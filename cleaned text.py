import pandas as pd
import re
import csv

df = pd.read_csv(r"C:\Users\Vijay\Desktop\infosys\mile1.csv", dtype=str)

def clean_text(x):
    if pd.isnull(x) or x == "":
        return ""
    s = str(x)
    s = re.sub(r"<[^>]*>", "", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def tokenize_text(x):
    if not x:
        return ""
    return "|".join(re.findall(r"\w+", x))

text_cols = df.select_dtypes(include=["object"]).columns

for col in text_cols:
    df[col] = df[col].map(clean_text)
    df[col + "_tokens"] = df[col].map(tokenize_text)

df = df.fillna("")

df.to_csv(r"C:\Users\Vijay\Desktop\mile1_cleaned.csv", index=False, encoding="utf-8-sig")