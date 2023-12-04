import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
dataset_path = 'Data/Fertilizer Prediction.csv'
data = pd.read_csv(dataset_path)

# Rename columns
data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

# Initialize LabelEncoder
encode_soil = LabelEncoder()
encode_crop = LabelEncoder()


# Fit and transform the categorical features
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)



# Separate the features and labels
features = data[['Nitrogen', 'Potassium', 'Phosphorous', 'Temperature', 'Humidity', 'Moisture', 'Crop_Type', 'Soil_Type']]
labels = data['Fertilizer']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Calculate accuracy
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

model_path = 'utils/Fertilizer_Prediction.pkl'
# Dump the model using pickle
with open(model_path, 'wb') as file:
    pickle.dump(classifier, file)
