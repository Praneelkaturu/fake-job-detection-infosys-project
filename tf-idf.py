import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

def clean_text_textcols(text):
    if pd.isnull(text):
        return text
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return tokens


def clean_text_structured(text):
    if pd.isnull(text):
        return text
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


input_path = r"C:\Users\Vijay\Desktop\infosys\mile1.csv"
df = pd.read_csv(input_path, dtype=str)


text_columns = [
    "title",
    "employment_type",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(clean_text_textcols)


if "salary_range" in df.columns:
    df["salary_range"] = df["salary_range"].astype(str).apply(
        lambda x: re.sub(r"<.*?>", " ", x).strip() if pd.notnull(x) else x
    )


first_col = df.columns[0]
other_columns = [
    col for col in df.columns
    if col not in text_columns + ["salary_range"] and col != first_col
]

for col in other_columns:
    df[col] = df[col].astype(str).apply(clean_text_structured)
output_path = r"C:\Users\Vijay\cleaned_file.csv"
df.to_csv(output_path, index=False)

df = pd.read_csv(output_path)

df["title_length"] = df["title"].astype(str).apply(len)
df["desc_length"] = df["description"].astype(str).apply(len)
plt.figure(figsize=(6, 4))
sns.histplot(df["title_length"], kde=True)
plt.savefig(r"C:\Users\Vijay\Desktop\infosys\title_length.png")

plt.figure(figsize=(6, 4))
sns.histplot(df["desc_length"], kde=True)
plt.savefig(r"C:\Users\Vijay\Desktop\infosys\desc_length.png")

if "fraudulent" in df.columns:
    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["fraudulent"])
    plt.savefig(r"C:\Users\Vijay\Desktop\infosys\fraudulent_dist.png")

text_all = " ".join(df["description"].astype(str))
wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig(r"C:\Users\Vijay\Desktop\infosys\wordcloud.png")

df["text_combined"] = (
    df["title"].astype(str) + " " +
    df["company_profile"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["requirements"].astype(str) + " " +
    df["benefits"].astype(str)
)
def fix_tokens(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            lst = ast.literal_eval(x)
            if isinstance(lst, list):
                return " ".join(map(str, lst))
        except:
            return x
    return x

df["text_combined"] = df["text_combined"].apply(fix_tokens)


tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X = tfidf.fit_transform(df["text_combined"])

y = df["fraudulent"].astype(int)

print("TF-IDF Matrix Shape:", X.shape)
print("Number of Features:", len(tfidf.get_feature_names_out()))
