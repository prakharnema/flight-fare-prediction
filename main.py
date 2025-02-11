from flask import Flask, render_template, request
import pickle
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load encoders and model
def load_pickle(file_path):
    """Load a pickle file."""
    try:
        return pickle.load(open(file_path, 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found. Ensure it exists and is accessible.")

class_encoder = load_pickle('class_encoder.pkl')
model = load_pickle('Flight_Fare_prediction_Model.pkl')
departure_time_encoder = load_pickle('departure_time_encoder.pkl')
arrival_time_encoder = load_pickle('arrival_time_encoder.pkl')

# Define unique values for dropdowns
unique_values = {
    'airline': ['Indigo', 'Air India', 'Vistara', 'GO_FIRST', 'AirAsia',
                'SpiceJet', 'Air India Express', 'AkasaAir'],
    'source_city': ['Kolkata', 'Mumbai', 'Hyderabad', 'Chennai', 'Delhi', 'Bangalore', 'Goa'],
    'destination_city': ['Bangalore', 'Delhi', 'Mumbai', 'Hyderabad', 'Chennai', 'Kolkata', 'Goa'],
    'class': ['Economy', 'Business'],
    'stops': ['zero', 'one', 'two_or_more']
}

# Map stops to numeric values
stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}

def safe_transform(encoder, value, default=0):
    """Transform a value using the encoder. Return a default value if the value is invalid."""
    return encoder.transform([value])[0] if value in encoder.classes_ else default

@app.route('/')
def index():
    """Render the home page with dropdown options and current date."""
    current_date = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
    return render_template('index.html', unique_values=unique_values, current_date=current_date)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        # Extract form data
        form_data = request.form
        airline = form_data['airline']
        source_city = form_data['source_city']
        destination_city = form_data['destination_city']
        departure_date = form_data['departure_date']
        departure_time = form_data['departure_time']
        arrival_time = form_data['arrival_time']
        travel_class = form_data['class']
        stops = form_data['stops']

        # Check if source and destination are the same
        if source_city == destination_city:
            return render_template('prediction_result.html', prediction="No such flight exists.")
        
        # Validate and parse date and time inputs
        departure_datetime = datetime.strptime(f"{departure_date} {departure_time}", '%Y-%m-%d %H:%M')
        arrival_datetime = datetime.strptime(f"{departure_date} {arrival_time}", '%Y-%m-%d %H:%M')
        
        def convert_time_to_period(time_input):
            """
            Converts the input time in HH:MM format into a period of the day.

            Args:
                time_input (str): Time in HH:MM format.

            Returns:
                str: The period of the day.
            """
            # Split the input into hours and minutes
            hours, minutes = map(int, time_input.split(":"))

            if 4 <= hours < 8:
                return "Early_Morning"
            elif 8 <= hours < 12:
                return "Morning"
            elif 12 <= hours < 16:
                return "Afternoon"
            elif 16 <= hours < 19:
                return "Evening"
            elif 19 <= hours < 23:
                return "Night"
            else:
                return "Late_Night"

        # Example usage
        # time_input = input("Enter the time (HH:MM): ")
        # period = convert_time_to_period(time_input)
        # print(f"The time {time_input} falls in the period: {period}")

        time_category_departure = convert_time_to_period(departure_time)
        departure_time_encoded = safe_transform(departure_time_encoder, time_category_departure)

        time_category_arrival = convert_time_to_period(arrival_time)
        arrival_time_encoded= safe_transform(arrival_time_encoder, time_category_arrival)
        duration = (arrival_datetime - departure_datetime).total_seconds() / 60
        days_left = (departure_datetime.date() - datetime.now().date()).days

        if days_left < 0:
            return render_template('prediction_result.html', prediction="Error: Departure date must be in the future.")

        # Encode categorical features
        departure_time_encoded = safe_transform(departure_time_encoder, departure_time)
        arrival_time_encoded = safe_transform(arrival_time_encoder, arrival_time)
        travel_class_encoded = class_encoder.transform([travel_class])[0]

        # Prepare input features (adjust to match the model's 25 features)
        input_features = np.zeros(25)
        input_features[:9] = [
            unique_values['airline'].index(airline),
            unique_values['source_city'].index(source_city),
            unique_values['destination_city'].index(destination_city),
            stops_mapping[stops],
            travel_class_encoded,
            duration,
            days_left,
            departure_time_encoded,
            arrival_time_encoded
        ]

        # Predict fare
        prediction = model.predict([input_features])[0]

        return render_template('prediction_result.html', prediction=f"Estimated Fare: ₹{prediction:.2f}")
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)