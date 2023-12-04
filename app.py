from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the crop prediction model
crop_prediction_model_path = 'utils/crop_prediction.pkl'
with open(crop_prediction_model_path, 'rb') as file:
    crop_prediction_model = pickle.load(file)

# Load the fertilizer prediction model
fertilizer_prediction_model_path = 'utils/Fertilizer_Prediction.pkl'
with open(fertilizer_prediction_model_path, 'rb') as file:
    fertilizer_prediction_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crop.html')
def crop():
    return render_template('crop.html')


@app.route('/fertilizer.html')
def fertilizer():
    return render_template('fertilizer.html')


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Prediction'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        temperature = int(request.form['temperature'])
        humidity = int(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create the input feature array for prediction
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make predictions using the loaded model
        prediction = crop_prediction_model.predict(input_features)

        return render_template('crop-result.html', prediction=prediction[0], title=title)


@app.route('/fertilizer-result', methods=['POST'])
def fertilizer_result():
    # Get the input values from the form
    N = int(request.form['inputN'])
    P = float(request.form['inputP'])
    K = float(request.form['inputK'])
    temperature = float(request.form['inputTemp'])
    humidity = float(request.form['inputHumidity'])
    moisture = float(request.form['inputMoisture'])
    crop_type = request.form['selectCropType']
    soil_type = request.form['selectSoilType']

    # Convert the categorical values to numerical values
    crop_type_encoded = 0
    soil_type_encoded = 0

    if crop_type == 'Barley':
        crop_type_encoded = 0
    elif crop_type == 'Cotton':
        crop_type_encoded = 1
    elif crop_type == 'Ground Nuts':
        crop_type_encoded = 2
    elif crop_type == 'Maize':
        crop_type_encoded = 3
    elif crop_type == 'Millets':
        crop_type_encoded = 4
    elif crop_type == 'Oil seeds':
        crop_type_encoded = 5
    elif crop_type == 'Paddy':
        crop_type_encoded = 6
    elif crop_type == 'Pulses':
        crop_type_encoded = 7
    elif crop_type == 'Sugarcane':
        crop_type_encoded = 8
    elif crop_type == 'Tobacco':
        crop_type_encoded = 9

    if soil_type == 'Black':
        soil_type_encoded = 0
    elif soil_type == 'Clayey':
        soil_type_encoded = 1
    elif soil_type == 'Loamy':
        soil_type_encoded = 2
    elif soil_type == 'Red':
        soil_type_encoded = 3
    elif soil_type == 'Sandy':
        soil_type_encoded = 4

    # Create the input feature array for prediction
    input_features = np.array([[N, P, K, temperature, humidity, moisture, crop_type_encoded, soil_type_encoded]])

    # Make predictions using the loaded model
    prediction = fertilizer_prediction_model.predict(input_features)

    # Render the result template with the predicted fertilizer type
    return render_template('fertilizer-result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
