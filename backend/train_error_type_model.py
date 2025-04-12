import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

def load_dataset(path="/Users/kaushiksai/Desktop/ai-python-debugger/backend/cleaned_python_only_dataset.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns={"original_src": "code", "error": "label"})

    # Drop nulls
    df.dropna(subset=["code", "label"], inplace=True)

    # Remove rare labels
    label_counts = df["label"].value_counts()
    df = df[df["label"].isin(label_counts[label_counts >= 3].index)]

    return df

def preprocess_data(df):
    print("Preprocessing data...")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    # Vectorize code
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["code"])

    # SMOTE to balance classes
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled, vectorizer, label_map

def train_models(X_train, y_train):
    print("Training models...")
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf

def evaluate_models(lr, rf, X_test, y_test, label_map):
    print("Evaluating models...")

    lr_probs = lr.predict_proba(X_test)
    rf_probs = rf.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=list(range(len(label_map))))

    fpr_lr, tpr_lr, _ = roc_curve(y_bin.ravel(), lr_probs.ravel())
    fpr_rf, tpr_rf, _ = roc_curve(y_bin.ravel(), rf_probs.ravel())

    auc_lr = roc_auc_score(y_bin, lr_probs, average='macro')
    auc_rf = roc_auc_score(y_bin, rf_probs, average='macro')

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/roc_curve.png")

    # Print classification reports
    print("\nLogistic Regression Report:")
    print(classification_report(y_test, lr.predict(X_test), target_names=label_map.values()))

    print("\nRandom Forest Report:")
    print(classification_report(y_test, rf.predict(X_test), target_names=label_map.values()))

def save_artifacts(lr, rf, vectorizer, label_map):
    print("Saving model and preprocessing artifacts...")
    os.makedirs("models", exist_ok=True)
    pickle.dump(lr, open("models/lr_model.pkl", "wb"))
    pickle.dump(rf, open("models/debugger_model.pkl", "wb"))  # main model
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(label_map, open("models/label_mapping.pkl", "wb"))

def main():
    print("Loading dataset...")
    df = load_dataset()
    X, y, vectorizer, label_map = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr, rf = train_models(X_train, y_train)
    evaluate_models(lr, rf, X_test, y_test, label_map)
    save_artifacts(lr, rf, vectorizer, label_map)

if __name__ == "__main__":
    main()
