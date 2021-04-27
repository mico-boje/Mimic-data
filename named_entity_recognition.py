from danlp.models.ner_taggers import load_flair_ner_model
from flair.data import Sentence

import pandas as pd
import re


# -

def clean_note(note):
    note = re.sub(r'\n', ' ', note)
    note = re.sub(r'[ ](?=[ ])|[^_,A-Za-z0-9æøåÆØÅ ]+', '', note)
    note = note.strip(" ")
    return note


# +
df = (pd.read_parquet('health_condition.parquet')
      .dropna(subset=['severity_code', "code_code"])
      .assign(note_len=lambda x: x['severity_code'].str.split(" ").apply(len))
      .assign(severity_code=lambda x: x['severity_code'].map(clean_note))
      .loc[:, ["code_text", "severity_code", "code_code", "parent_code", "note_len"]])

# Exclude non-sensical notes
exclude_codes = ["", ".", "xxx", "xx", "x", "..", "fejl", "Oprettelse"]
df = df[~df["severity_code"].isin(exclude_codes)]

df["severity_code"].tolist()[:10]
# -

# Load the NER tagger using the DaNLP wrapper
flair_model = load_flair_ner_model()

# Using the flair NER tagger
sentence = Sentence(df["severity_code"].iloc[5]) 
flair_model.predict(sentence) 
print(sentence.to_tagged_string())