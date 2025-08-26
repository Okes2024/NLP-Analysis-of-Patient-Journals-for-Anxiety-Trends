# nlp_anxiety_trends.py
"""
NLP Analysis of Patient Journals for Anxiety Trends (Synthetic Data)

- Generates synthetic journal entries for multiple patients.
- Computes a simple lexicon-based anxiety score.
- Extracts features (TF-IDF) and fits LDA topics.
- Plots population-level and per-patient anxiety trends.
- Saves the synthetic dataset to Excel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

rng = np.random.default_rng(42)

# -------------------------
# Synthetic data generation
# -------------------------
n_patients = 25
entries_per_patient = rng.integers(6, 12, size=n_patients)  # 6–11 entries each
patient_ids = [f"P{str(i+1).zfill(3)}" for i in range(n_patients)]

dates = pd.date_range("2025-01-01", "2025-04-30", freq="D")
base_anxiety_by_patient = {pid: rng.uniform(0.2, 0.8) for pid in patient_ids}

anxiety_words = [
    "anxious","worry","panic","fear","nervous","restless","tense","uneasy",
    "racing","overwhelmed","sweat","shaky","avoid","ruminate","catastrophize",
    "trouble","insomnia","dread","stress","tight","nauseous","palpitations",
]
neutral_words = [
    "work","school","family","friend","walk","coffee","breakfast","meeting",
    "project","movie","music","weather","traffic","study","book","exercise",
    "dinner","garden","cook","clean","travel","call","message","plan",
]
coping_words = [
    "breathe","journal","therapy","meditate","mindful","relax","yoga","walked",
    "talked","coping","grounding","progress","support","improve","sleep","rest",
]

def synth_entry(pid, date):
    base = base_anxiety_by_patient[pid]
    weekday_factor = 1.15 if date.weekday() in (0,1) else (0.95 if date.weekday()>=5 else 1.0)
    shock = rng.normal(0, 0.08)
    level = np.clip(base * weekday_factor + shock, 0, 1)
    n_anx = int(1 + rng.poisson(2 + 6*level))
    n_neu = int(5 + rng.poisson(10 - 5*level))
    n_cop = int(1 + rng.poisson(2 + 3*(1-level)))
    words = rng.choice(anxiety_words, n_anx).tolist()               + rng.choice(neutral_words, n_neu).tolist()               + rng.choice(coping_words, n_cop).tolist()
    rng.shuffle(words)
    # Simple sentence assembly
    sents = []
    for i in range(max(3, int(len(words)/7))):
        chunk = words[i*7:(i+1)*7]
        if not chunk: break
        sents.append(" ".join(chunk).capitalize() + ".")
    text = " ".join(sents)
    score = (n_anx - 0.3*n_cop) / max(10, (n_anx+n_neu+n_cop))
    score = float(np.clip(score, 0, 1))
    return text, score, n_anx, n_neu, n_cop

rows = []
for pid, k in zip(patient_ids, entries_per_patient):
    day_idxs = sorted(rng.choice(len(dates), size=int(k), replace=False).tolist())
    for di in day_idxs:
        date = dates[di]
        text, score, n_anx, n_neu, n_cop = synth_entry(pid, date)
        rows.append({
            "patient_id": pid,
            "date": date,
            "journal_text": text,
            "anxiety_score": score,
            "tokens_anxiety": n_anx,
            "tokens_neutral": n_neu,
            "tokens_coping": n_cop,
        })

df = pd.DataFrame(rows).sort_values(["patient_id","date"]).reset_index(drop=True)

# Save dataset
df.to_excel("patient_journals_anxiety_synthetic.xlsx", index=False)
print(f"Saved dataset with {len(df)} rows to patient_journals_anxiety_synthetic.xlsx")

# -------------------------
# NLP Feature Extraction (TF-IDF) and Topic Modeling (LDA)
# -------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.9,
    min_df=2,
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(df["journal_text"].values)

n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method="batch")
topic_weights = lda.fit_transform(X)
df_topics = pd.DataFrame(topic_weights, columns=[f"topic_{i}" for i in range(n_topics)])
df = pd.concat([df, df_topics], axis=1)

# Save topic top-terms
def top_terms_for_topic(model, feature_names, k=10):
    comp = model.components_
    tops = {}
    for t, row in enumerate(comp):
        idx = np.argsort(row)[::-1][:k]
        tops[f"topic_{t}"] = [feature_names[i] for i in idx]
    return pd.DataFrame.from_dict(tops, orient="index").T

feature_names = np.array(vectorizer.get_feature_names_out())
top_terms_df = top_terms_for_topic(lda, feature_names, k=10)
top_terms_df.to_csv("lda_topic_top_terms.csv", index=False)

# -------------------------
# Trend Analysis
# -------------------------
df["date"] = pd.to_datetime(df["date"])
daily = df.groupby("date")["anxiety_score"].mean().reset_index(name="mean_anxiety")
per_patient = (
    df.sort_values(["patient_id","date"])
      .groupby("patient_id")
      .apply(lambda g: g.assign(rolling7=g["anxiety_score"].rolling(3, min_periods=1).mean()))
      .reset_index(drop=True)
)
per_patient.to_csv("per_patient_anxiety_trend.csv", index=False)

# -------------------------
# Visualizations (matplotlib only, one chart per figure, no explicit colors)
# -------------------------
plt.figure(figsize=(9,4))
plt.plot(daily["date"], daily["mean_anxiety"])
plt.title("Population Mean Anxiety Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Anxiety Score")
plt.tight_layout()
plt.show()

# Example patient plot
example_id = df["patient_id"].iloc[0]
eg = per_patient[per_patient["patient_id"] == example_id]
plt.figure(figsize=(9,4))
plt.plot(eg["date"], eg["anxiety_score"], marker="o")
plt.plot(eg["date"], eg["rolling7"])
plt.title(f"Anxiety Trend for {example_id}")
plt.xlabel("Date")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# Topic weight distribution
plt.figure(figsize=(6,4))
plt.hist(topic_weights.argmax(axis=1), bins=np.arange(5)-0.5, rwidth=0.8)
plt.title("Dominant Topic Distribution")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("Saved: lda_topic_top_terms.csv, per_patient_anxiety_trend.csv")
