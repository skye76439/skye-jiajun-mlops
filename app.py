from flask import Flask, render_template, request
import pandas as pd
from pycaret.regression import predict_model, load_model

app = Flask(__name__)

# Load the trained model
model_path = "models/skye/tuned_model_for_price_prediction"
model = load_model(model_path)

# List of features expected by the model
all_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'availability_30', 'bed_type', 'calculated_host_listings_count', 'cancellation_policy',
    'guests_included', 'has_availability', 'host_is_superhost', 'host_listings_count',
    'instant_bookable', 'latitude(North)', 'longitude(East)', 'maximum_nights',
    'number_of_reviews', 'property_type', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_rating',
    'review_scores_value', 'room_type'
]

# Initialize default values for all features
default_input = {feature: 0 for feature in all_features}

@app.route('/')
def index():
    return render_template('skye_prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input from the form
    user_input = {}
    for feature in all_features:
        user_input[feature] = request.form.get(feature, default_input[feature])

    # Convert user input into a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Use the model to predict
    predictions = predict_model(model, data=user_input_df)

    # Extract the predicted price
    predicted_price = round(predictions['prediction_label'].iloc[0], 2)

    # Pass the predicted price to the template
    return render_template('skye_prediction.html', predicted_price=predicted_price)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
