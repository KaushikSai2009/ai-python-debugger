import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from ai_model import AIModel

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv('/Users/kaushiksai/Desktop/ai-python-debugger/backend/cleaned_python_only_dataset.csv')
    
    # Handle missing values
    data = data.dropna()
    
    # Split features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle class imbalance
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Dimensionality reduction
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_resampled)
    
    return train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42)

def main():
    X_train, X_test, y_train, y_test = preprocess_data('data.csv')
    model = AIModel()
    
    # Train and evaluate models
    for model_name in model.models.keys():
        model.train(model_name, X_train, y_train)
        scores = model.evaluate(model_name, X_test, y_test)
        print(f"{model_name} Scores: {scores}")

if __name__ == "__main__":
    main()