# Import necessary libraries
import pandas as pd
import numpy as np  # Import numpy for numerical operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    print("Data Loaded:")
    print(df.head())  # Print the first few rows of the DataFrame
    return df  # Return the DataFrame

# Preprocess the data
from sklearn.preprocessing import LabelEncoder

# Preprocess the data
def preprocess_data(df):
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (the last column)

    # Handle categorical variables (convert to numeric)
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Handle missing values (if any)
    X = X.fillna(X.mean())  # You could also use median or another strategy

    # Standardize the features (scale them)
    scaler = StandardScaler()  # Create a scaler object
    X_scaled = scaler.fit_transform(X)  # Fit the scaler and transform the features
    
    return X_scaled, y  # Return the scaled features and target variable


# Train the model
def train_model(X, y):
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's accuracy
    print("Model Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy score
    
    return model  # Return the trained model

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)  # Save the model to a file

# Main function to execute the steps
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('indian.csv')  # Load the dataset
    X, y = preprocess_data(data)  # Preprocess the data
    
    # Train the model
    model = train_model(X, y)  # Train the model
    
    # Save the trained model
    save_model(model, 'liver_model.pkl')  # Save the model to a file
