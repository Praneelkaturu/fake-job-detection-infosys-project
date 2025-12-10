import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

df = pd.read_csv(r"C:\Users\Vijay\Desktop\infosys\mile1_cleaned.csv", dtype=str)
df = df.fillna("")
df["fraudulent"] = df["fraudulent"].astype(int)

def tokenize(s):
    if not isinstance(s, str):
        return []
    return re.findall(r"\w+", s)

df["desc_length"] = df["description"].apply(lambda x: len(tokenize(x)))
df["title_length"] = df["title"].apply(lambda x: len(tokenize(x)))

real_text = " ".join(df[df["fraudulent"] == 0]["description"])
fake_text = " ".join(df[df["fraudulent"] == 1]["description"])

wc_real = WordCloud(width=800, height=400).generate(real_text)
wc_fake = WordCloud(width=800, height=400).generate(fake_text)

plt.figure(figsize=(10,5))
plt.title("Real Job Wordcloud")
plt.imshow(wc_real)
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.title("Fake Job Wordcloud")
plt.imshow(wc_fake)
plt.axis("off")
plt.show()

desc_counts = df["desc_length"].value_counts().sort_index()
plt.figure(figsize=(7,4))
plt.bar(desc_counts.index, desc_counts.values)
plt.title("Description Length Distribution")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.show()

emp_counts = df["employment_type"].value_counts()
plt.figure(figsize=(8,4))
plt.bar(emp_counts.index, emp_counts.values)
plt.title("Employment Type Distribution")
plt.xlabel("Employment Type")
plt.ylabel("Count")
plt.show()

fake = df[df["fraudulent"] == 1]
real = df[df["fraudulent"] == 0]

print("Real Job Stats:\n", real.describe(include='all'))
print("Fake Job Stats:\n", fake.describe(include='all'))