import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
# Load the dataset
# File paths
dataset_path = 'data/crop_recommendation.csv'

# Load the dataset
data = pd.read_csv(dataset_path)

# Split the data into input features (X) and target variable (y)
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier
crop_prediction_model = GaussianNB()

# Train the model
crop_prediction_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = crop_prediction_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

model_path = 'utils\crop_prediction.pkl'
# Dump the model using pickle
with open(model_path, 'wb') as file:
    pickle.dump(crop_prediction_model, file)
