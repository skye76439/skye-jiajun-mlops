from flask import Flask, render_template, request
import pandas as pd
from pycaret.classification import predict_model, load_model
from pycaret.regression import predict_model as predict_regression
import os

app = Flask(__name__)

# Function to load the Jia Jun model
def load_jiajun_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "jiajun", "final_model_mushroom")
    return load_model(model_path)

# Function to load the Skye model
def load_skye_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "skye", "tuned_model_for_price_prediction")
    return load_model(model_path)

# Load Jia Jun model
jiajun_model = load_jiajun_model()

# Load Skye model
skye_model = load_skye_model()

# List of features expected by the Skye model
all_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'availability_30', 'bed_type', 'calculated_host_listings_count', 'cancellation_policy',
    'guests_included', 'has_availability', 'host_is_superhost', 'host_listings_count',
    'instant_bookable', 'latitude(North)', 'longitude(East)', 'maximum_nights',
    'number_of_reviews', 'property_type', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_rating',
    'review_scores_value', 'room_type'
]

# Initialize default values for all features for Skye Prediction
default_input_skye = {feature: 0 for feature in all_features}

@app.route('/', methods=['GET', 'POST'])
def predict_jiajun():
    if request.method == 'POST':
        try:
            user_input = {
                'cap-shape': request.form['cap-shape'],
                'cap-surface': request.form['cap-surface'],
                'cap-color': request.form['cap-color'],
                'bruises': request.form['bruises'],
                'odor': request.form['odor']
            }
            user_input_df = pd.DataFrame([user_input])
            predictions = predict_model(jiajun_model, data=user_input_df)
            prediction = predictions['prediction_label'].iloc[0]
            score = predictions['prediction_score'].iloc[0]
            return render_template('jiajun_classification.html', prediction=prediction, score=score)
        except Exception as e:
            return render_template('jiajun_classification.html', error=str(e))

    return render_template('jiajun_classification.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_skye():
    if request.method == 'POST':
        try:
            # Collect user input from the form
            user_input = {}
            for feature in all_features:
                user_input[feature] = float(request.form.get(feature, default_input_skye[feature]))

            # Convert user input into a DataFrame
            user_input_df = pd.DataFrame([user_input])

            # Use the model to predict
            predictions = predict_regression(skye_model, data=user_input_df)

            # Extract the predicted price
            predicted_price = round(predictions['prediction_label'].iloc[0], 2)

            # Pass the predicted price to the template
            return render_template('skye_prediction.html', predicted_price=predicted_price)
        except Exception as e:
            return render_template('skye_prediction.html', error=str(e))

    return render_template('skye_prediction.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
