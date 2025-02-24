import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class HealthPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = [
            'Age', 'sex', 'cp', 'trest', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
    def load_data(self, path='Data.csv'):
        """Load and prepare data"""
        print("Loading data...")
        data = pd.read_csv(path)
        
        # Convert categorical data
        categorical_cols = ['Age', 'trest', 'chol', 'thalach', 'oldpeak']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col])
        
        return data
    
    def train_model(self):
        """Train the prediction model"""
        print("\nTraining model...")
        
        # Load data
        data = self.load_data()
        
        # Split features and target
        X = data[self.features]
        y = data['target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nTraining Results:")
        print(f"- Training accuracy: {train_score:.2f}")
        print(f"- Testing accuracy: {test_score:.2f}")
        
        # Save model
        joblib.dump(self, 'health_model.joblib')
        print("\nModel saved as: health_model.joblib")
    
    def predict(self):
        """Input information and make prediction"""
        print("\nENTER PATIENT INFORMATION:")
        print("=========================")
        
        # Input guidelines
        prompts = {
            'Age': 'Age (Adult/MidleAge/Old): ',
            'sex': 'Gender (0: Female, 1: Male): ',
            'cp': 'Chest pain type (0-3): ',
            'trest': 'Resting blood pressure (Normal/High/Low): ',
            'chol': 'Cholesterol (Normal/High Risk/Extreme): ',
            'fbs': 'Fasting blood sugar > 120 mg/dl (0: No, 1: Yes): ',
            'restecg': 'Resting ECG results (0-2): ',
            'thalach': 'Maximum heart rate (Normal/High/Low): ',
            'exang': 'Exercise induced angina (0: No, 1: Yes): ',
            'oldpeak': 'ST depression (High/Normal/Low): ',
            'slope': 'ST peak slope (0-2): ',
            'ca': 'Number of major vessels (0-4): ',
            'thal': 'Thalassemia (1-3): '
        }
        
        # Get input data
        input_data = {}
        for name, prompt in prompts.items():
            while True:
                value = input(prompt)
                if name in self.label_encoders:
                    try:
                        # Convert categorical values
                        value = self.label_encoders[name].transform([value])[0]
                        break
                    except:
                        print("Invalid value! Please try again.")
                else:
                    try:
                        value = int(value)
                        break
                    except:
                        print("Please enter a number!")
            input_data[name] = value
        
        # Make prediction
        input_df = pd.DataFrame([input_data])
        result = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Print results
        print("\nPREDICTION RESULTS:")
        print("==================")
        print(f"Status: {'Healthy' if result == 1 else 'Not Healthy'}")
        print(f"Probability of being unhealthy: {probabilities[0]:.2f}")
        print(f"Probability of being healthy: {probabilities[1]:.2f}")

def main():
    # Create predictor object
    predictor = HealthPredictor()
    
    while True:
        print("\nHEART HEALTH PREDICTION SYSTEM")
        print("==============================")
        print("1. Train new model")
        print("2. Make prediction")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ")
        
        if choice == '1':
            predictor.train_model()
        elif choice == '2':
            # Check if model exists
            if predictor.model is None:
                try:
                    # Try to load saved model
                    predictor = joblib.load('health_model.joblib')
                except:
                    print("\nNo model found! Please train a model first.")
                    continue
            predictor.predict()
        elif choice == '3':
            print("\nThank you for using the program!")
            break
        else:
            print("\nInvalid choice!")

if __name__ == "__main__":
    main() 