import os
import re
import json
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from pathlib import Path
from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertConfig, BertTokenizer, TFBertModel, AutoTokenizer

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import altair as alt
# -

# ### Setup TF

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})


# ### Load data

def clean_note(note):
    note = re.sub(r'\n', ' ', note)
    note = re.sub(r'[ ](?=[ ])|[^_,A-Za-z0-9æøåÆØÅ ]+', '', note)
    note = note.strip(" ")
    return note


# +
#cleanr = re.compile(r'[ ](?=[ ])|[^_,A-Za-z0-9 ]+')
bert_model_path = "/data/cura/bert_models/danish_bert_uncased_v2"
df = (pd.read_parquet('health_condition.parquet')
      .dropna(subset=['severity_code', "code_code"])
      .assign(note_len=lambda x: x['severity_code'].str.split(" ").apply(len))
      .assign(severity_code=lambda x: x['severity_code'].map(clean_note))
      .loc[:, ["code_text", "severity_code", "code_code", "parent_code", "note_len"]])

# Exclude non-sensical notes
exclude_codes = ["", ".", "xxx", "xx", "x", "..", "fejl", "Oprettelse"]
df = df[~df["severity_code"].isin(exclude_codes)]

df.head()
# -

df.shape

# Note length distribution

df['note_len'].plot.hist(bins=np.arange(0, 60, 2));

# Most prevalent notes of a single word

(df[df["note_len"] < 2]['severity_code']
 .astype(str)
 .value_counts()
 .sort_values(ascending=False)
 .loc[lambda x: x.quantile(.9) < x]
 .plot.bar(figsize=(15, 5)));

# Note frequency distribution

df["severity_code"].value_counts().plot.hist(logy=True, bins=np.arange(0, 30, 1));

# Most prevalent notes

df["severity_code"].value_counts().iloc[:10]

# ### Prepare data for training

# +
TEXT_COL   = 'severity_code'
TARGET_COL = 'parent_code'

train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(df[TEXT_COL].tolist(), 
                                                                          df[TARGET_COL].cat.codes, 
                                                                          random_state=0, 
                                                                          test_size=0.1,
                                                                          stratify=df[TARGET_COL].cat.codes)
train_size = len(train_inputs)
valid_size = len(valid_inputs)
NUM_LABELS = df[TARGET_COL].nunique()
BATCH_SIZE = 32
EPOCHS = 20
MAX_SEQ_LEN = 50
train_steps = train_size // BATCH_SIZE
valid_steps = valid_size // BATCH_SIZE
# -

with open(bert_model_path + "/bert_config.json", "r") as f:
    print(json.load(f))

train_inputs[:5]

train_labels[:5]


# ### Tokenize sentence

def tokenize_sentences(sentences, tokenizer, max_seq_len = 128):
    return tokenizer.batch_encode_plus(
        sentences, 
        max_length = max_seq_len, 
        pad_to_max_length=True,
        return_token_type_ids=False,
        add_special_tokens=True)


tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)

tokenizer.tokenize(train_inputs[4])

train_ids = tokenize_sentences(train_inputs, tokenizer, MAX_SEQ_LEN)
valid_ids = tokenize_sentences(valid_inputs, tokenizer, MAX_SEQ_LEN)

train_ids['input_ids'][4][:12]

train_ids.keys()


# ### Transform to tf.data

def create_dataset(inputs, labels):
    def gen():
        for i in range(len(labels)):
            yield (
                {
                    "input_ids": inputs["input_ids"][i],
                    "attention_mask": inputs["attention_mask"][i],
#                    "token_type_ids": inputs["token_type_ids"][i],
                },
                labels[i],
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
#                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


# +
train_dataset = create_dataset(train_ids, train_labels.tolist())
valid_dataset = create_dataset(valid_ids, valid_labels.tolist())

train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(train_steps*EPOCHS)
valid_dataset = valid_dataset.batch(BATCH_SIZE).repeat(valid_steps*EPOCHS)


# -

# ### Define model

class BertModel(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert", trainable=False)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), 
            name="classifier"
        )

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        #sequence_output = self.dropout(outputs[0], training=kwargs.get("training", False))
        pooled_output = self.dropout(outputs[1], training=kwargs.get("training", False))
        logits = self.classifier(pooled_output)
        
        return logits


# ### Train

config = BertConfig.from_pretrained(bert_model_path + '/bert_config.json', num_labels=NUM_LABELS, max_length=MAX_SEQ_LEN)
model = BertModel(config=config)
model.load_weights(bert_model_path + "/bert_model.ckpt.index");

# + active=""
# bert_model = TFBertModel(config)
# bert_model.load_weights(bert_model_path + "/bert_model.ckpt.index");

# +
# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (5 - epoch))
#lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[lr_callback],
    steps_per_epoch=train_steps, # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
    validation_data=valid_dataset,
    validation_steps=valid_steps,
)
# -

# #### Save

# + active=""
# os.makedirs("./save/", exist_ok=True)
# model.save_pretrained("./save/")
# -

# ### Evaluate

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left');

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left');

# Get predictions

predict_dataset = create_dataset(valid_ids, valid_labels.tolist()).batch(BATCH_SIZE)

valid_predictions = list()
for inputs, labels in iter(predict_dataset):
    valid_predictions.append(tf.argmax(model(inputs), axis=1).numpy())
valid_predictions = np.hstack(valid_predictions)

# Calculate accuracy

np.mean(valid_predictions == valid_labels)

# Evaluate results

result = df.loc[valid_labels.index].copy().assign(correct=(valid_predictions == valid_labels))
result.head()

result["correct"].mean()

alt.Chart(result).mark_bar().encode(
    x=alt.X("count(correct):O"),
    y=alt.Y("code_text:N"),
    color="correct",
)