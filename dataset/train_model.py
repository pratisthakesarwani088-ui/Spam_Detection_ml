import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Dataset load
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)



# Data cleaning
df["message"] = df["message"].str.lower()

df["message"] = df["message"].str.replace(
    r"[^a-zA-Z0-9\s]", "", regex=True
)

df["message"] = df["message"].str.strip()


# Input and output
X = df["message"]
y = df["label"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Naive Bayes Model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# Prediction
y_pred = model.predict(X_test_tfidf)

print(y_pred[:10])
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:", accuracy)
with open("model/accuracy.txt","w") as f:
    f.write(str(accuracy))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
print("confusion matrix")
print(confusion_matrix(y_test,y_pred))
message=["Congratulation! you are pass"]
message_tfidf=vectorizer.transform(message)
prediction=model.predict(message_tfidf)
print("Prediction:",prediction[0])
joblib.dump(model,"model/spam_model.pkl")
joblib.dump(vectorizer,"model/tfidf_vectorizer.pkl")
print("Model saved successfully!")