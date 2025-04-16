import pickle
from sklearn.ensemble import RandomForestClassifier

# Example training process
X_train = [[5, 2], [15, 6], [25, 10]]  # Dummy feature vectors
y_train = ["Syntax Error", "Indentation Error", "Name Error"]
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("error_type_model.pkl", "wb") as f:
    pickle.dump(model, f)