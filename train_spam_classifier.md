#!/usr/bin/env python3 \"\"\" train_spam_classifier.py

\- If task_3/spam.csv does not exist, this script generates a small
synthetic dataset. - Trains a TF-IDF + LogisticRegression spam
classifier. - Saves artifacts to task_3/model/ - Produces evaluation
plots. \"\"\"

from pathlib import Path import random import csv import os

\# \-\-- Dataset generation (only if spam.csv missing) \-\-- BASE =
Path(\_\_file\_\_).resolve().parent CSV_PATH = BASE / \"spam.csv\"
MODEL_DIR = BASE / \"model\" MODEL_DIR.mkdir(exist_ok=True)

def generate_dataset(path, n_extra=150, seed=42): random.seed(seed)
base_rows = \[ (\"Win a FREE ticket to Bahamas! Call now to claim your
prize.\", \"spam\"), (\"Lowest price for meds, no prescription needed.
Order today!\", \"spam\"), (\"Reminder: your dentist appointment is
tomorrow at 10:30 AM.\", \"ham\"), (\"Can you send me the revised budget
spreadsheet?\", \"ham\"), (\"URGENT! Your account has been compromised.
Verify immediately.\", \"spam\"), (\"Meeting notes from yesterday
attached.\", \"ham\"), (\"Congratulations --- you\'ve been selected for
a \$1000 gift card.\", \"spam\"), (\"Lunch at 1 pm? I\'ll bring the
files.\", \"ham\"), (\"Limited time offer: Get 50% off on all items.\",
\"spam\"), (\"Could you review the draft by noon?\", \"ham\") \]

spam_templates = \[ \"Congratulations! You won {amount} dollars. Claim
at {link}\", \"Dear user, your subscription will expire. Renew now at
{link}\", \"You have an unpaid invoice of {amount}. Click {link} to
pay\", \"Exclusive deal: {percent}% off on your next purchase. Visit
{link}\", \"Get access to our investment program and earn {amount}
weekly.\", \"FREE access to premium movies. Sign up at {link}\",
\"Lowest prices on meds. Buy {product} without prescription.\",
\"Increase followers fast --- only {amount} per day. Buy now!\", \"You
have been selected for a trial. Claim at {link}\", \]

ham_templates = \[ \"Can you review the document I sent earlier?\",
\"Let\'s meet at {place} around {time}.\", \"Please confirm your
availability for the meeting on {date}.\", \"Thanks for your help with
the {project}.\", \"I\'ve attached the files you requested.\",
\"Reminder: team standup at {time}.\", \"Can you share the contact
details for {person}?\", \"I\'ll be on leave on {date}.\", \"Please
update the spreadsheet with the latest numbers.\", \"Good job on the
presentation today!\" \]

links = \[\"http://bit.ly/offer\", \"http://free.prize/claim\",
\"http://verify.me/login\"\] products = \[\"viagra\", \"painkillers\",
\"no-prescription medicine\"\] places = \[\"the cafe\", \"conference
room B\", \"Zoom\"\] times = \[\"10:00 AM\", \"2:30 PM\", \"tomorrow
morning\"\] dates = \[\"June 10\", \"next Monday\", \"2025-07-01\"\]
persons = \[\"John\", \"Anna\", \"the supplier\"\] projects = \[\"Q3
migration\", \"budget review\", \"audit\"\]

rows = base_rows.copy() for \_ in range(n_extra): if random.random() \<
0.6: tmpl = random.choice(spam_templates) text =
tmpl.format(amount=f\"\${random.randint(50,5000)}\",
link=random.choice(links), percent=random.randint(10,90),
product=random.choice(products)) label = \"spam\" else: tmpl =
random.choice(ham_templates) text =
tmpl.format(place=random.choice(places), time=random.choice(times),
date=random.choice(dates), person=random.choice(persons),
project=random.choice(projects)) label = \"ham\" rows.append((text,
label))

random.shuffle(rows) with open(path, \"w\", newline=\"\",
encoding=\"utf-8\") as f: writer = csv.writer(f)
writer.writerow(\[\"text\", \"label\"\]) for t, l in rows:
writer.writerow(\[t, l\]) print(f\"Generated dataset with {len(rows)}
rows at {path}\")

if not CSV_PATH.exists(): generate_dataset(CSV_PATH)

\# \-\-- Training pipeline \-\-- import pandas as pd import numpy as np
from sklearn.model_selection import train_test_split from
sklearn.feature_extraction.text import TfidfVectorizer from
sklearn.linear_model import LogisticRegression from sklearn.metrics
import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib import matplotlib.pyplot as plt

def train_and_save(csv_path, model_dir): df = pd.read_csv(csv_path)
print(\"Dataset shape:\", df.shape) print(df.label.value_counts())

X = df.text.values y = np.where(df.label == \"spam\", 1, 0)

X_train, X_test, y_train, y_test = train_test_split( X, y,
test_size=0.2, random_state=42, stratify=y )

vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vect.fit_transform(X_train) X_test_tfidf =
vect.transform(X_test)

clf = LogisticRegression(max_iter=1000) clf.fit(X_train_tfidf, y_train)

\# Save artifacts joblib.dump(clf, model_dir / \"spam_clf.joblib\")
joblib.dump(vect, model_dir / \"tfidf_vectorizer.joblib\")

\# Evaluate y_pred = clf.predict(X_test_tfidf) report =
classification_report(y_test, y_pred, target_names=\[\"ham\",
\"spam\"\]) print(report) with open(model_dir /
\"classification_report.txt\", \"w\", encoding=\"utf-8\") as f:
f.write(report)

\# Confusion matrix plot cm = confusion_matrix(y_test, y_pred) disp =
ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=\[\"ham\",\"spam\"\]) fig, ax =
plt.subplots(figsize=(4,4)) disp.plot(ax=ax) plt.title(\"Confusion
Matrix\") plt.tight_layout() fig.savefig(model_dir /
\"confusion_matrix.png\") plt.close(fig)

\# Top features feature_names = np.array(vect.get_feature_names_out())
coefs = clf.coef\_\[0\] top_spam_idx = np.argsort(coefs)\[-20:\]\[::-1\]

fig, ax = plt.subplots(figsize=(8,6)) top_feats =
feature_names\[top_spam_idx\] top_vals = coefs\[top_spam_idx\]
ax.barh(range(len(top_feats))\[::-1\], top_vals\[::-1\])
ax.set_yticks(range(len(top_feats))) ax.set_yticklabels(top_feats)
ax.set_title(\"Top features indicative of SPAM\") plt.tight_layout()
fig.savefig(model_dir / \"top_spam_features.png\") plt.close(fig)

print(\"Saved model and visualizations to\", model_dir)

if \_\_name\_\_ == \"\_\_main\_\_\": train_and_save(CSV_PATH, MODEL_DIR)
