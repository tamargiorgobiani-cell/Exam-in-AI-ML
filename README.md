Task 3 --- Spam vs Non-Spam Classifier (User Guide)

Summary

This project trains a small machine learning model to distinguish spam
from non-spam messages. It is meant as an educational demo. The
implementation uses TF-IDF for text representation and Logistic
Regression for classification.

Files

\- \`train_spam_classifier.py\` --- main script. Generates dataset if
missing, trains model, saves artifacts, and creates plots. -
\`spam.csv\` --- dataset (generated automatically by the script if
absent). - \`model/\` --- created by the script; contains:  -
\`spam_clf.joblib\` --- trained classifier  -
\`tfidf_vectorizer.joblib\` --- TF-IDF vectorizer  -
\`confusion_matrix.png\` --- confusion matrix visualization  -
\`top_spam_features.png\` --- bar chart of top spam-indicative features
 - \`classification_report.txt\` --- precision/recall/f1 report -
\`train_output.log\` --- optional runtime log if you capture output.

Quickstart

1\. Create & activate virtual environment:

python3 -m venv venv source venv/bin/activate \# Windows:
venv\\\\Scripts\\\\activate pip install -U pip pip install pandas
scikit-learn matplotlib joblib

2\. Run training: python task_3/train_spam_classifier.py

3\. Inspect artifacts

ls task_3/model \# open images: task_3/model/confusion_matrix.png \#
open: task_3/model/top_spam_features.png \# open:
task_3/model/classification_report.txt

How to use the model for prediction (example) import joblib vect =
joblib.load(\"task_3/model/tfidf_vectorizer.joblib\") clf =
joblib.load(\"task_3/model/spam_clf.joblib\")

sample = \[\"Free prize! Click http://free.prize/claim to get it\"\] X =
vect.transform(sample) print(clf.predict(X)) \# 1 -\> spam, 0 -\> ham

Visualizations --- what they show

Confusion Matrix: rows = true labels, columns = predicted labels. Helps
you identify false positives (ham predicted as spam) and false negatives
(spam missed).

Top Spam Features: words/phrases with highest positive coefficients for
the spam class. Useful for interpreting what the model relies on.

Suggestions for improvement (beyond this demo)

Use a larger real dataset (e.g., SMS Spam Collection, Enron).

Apply text cleaning: lowercasing, URL removal, tokenization,
lemmatization.

Use cross-validation and hyperparameter tuning (GridSearchCV).

Try alternative models: RandomForest, XGBoost, or deep learning (LSTM,
Transformers).

Add adversarial robustness (spammers adapt wording).
