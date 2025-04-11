import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class AIModel:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(probability=True)
        }
        self.model_scores = {}

    def train(self, model_name, X_train, y_train):
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.models[model_name] = model

    def evaluate(self, model_name, X_test, y_test):
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        
        # Calculate scores
        scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        self.model_scores[model_name] = scores
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        self.plot_roc_curve(fpr, tpr, roc_auc, model_name)
        
        return scores

    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        plt.show()