from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("Electric_Vehicle_Sales.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/', methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template('index.html',form_data={})

    elif request.method == "POST":

        form_data = request.form.to_dict()

        # Get input values from HTML form
        try:
            battery_capacity = float(request.form.get('battery_capacity'))
            range_km = float(request.form.get('range_km'))
            price_lakhs = float(request.form.get('price_lakhs'))
            charging_time = float(request.form.get('charging_time'))
            brand_rating = float(request.form.get('brand_rating'))
            incentives = float(request.form.get('incentives'))
            showroom_locations = float(request.form.get('showroom_locations'))

        except (ValueError, TypeError):
            return render_template(
                'index.html',
                error="Please enter valid numerical values.",
                form_data=form_data
            )

        # Validate input values based on training dataset ranges
        if not (40 <= battery_capacity <= 96):
            error = "Battery Capacity must be between 40 and 96 kWh"

        elif not (250 <= range_km <= 520):
            error = "Range must be between 250 and 520 km"

        elif not (18.6 <= price_lakhs <= 50.4):
            error = "Price must be between 18.6 and 50.4 lakhs"

        elif not (6.5 <= charging_time <= 10.2):
            error = "Charging Time must be between 6.5 and 10.2 hours"

        elif not (7.3 <= brand_rating <= 9.4):
            error = "Brand Rating must be between 7.3 and 9.4"

        elif not (3 <= incentives <= 8):
            error = "Incentives must be between 3 and 8"

        elif not (6 <= showroom_locations <= 23):
            error = "Showroom Locations must be between 6 and 23"

        elif not showroom_locations.is_integer():
            error = "Showroom Locations must be a whole number"

        else:
            error = None

        # Display validation error on the input page
        if error:
            return render_template(
                'index.html',
                error=error,
                form_data=form_data
            )

        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'BatteryCapacity_kWh': battery_capacity,
            'Range_km': range_km,
            'Price_Lakhs': price_lakhs,
            'ChargingTime_hrs': charging_time,
            'BrandRating': brand_rating,
            'Incentives_kWh': incentives,
            'ShowroomLocations': showroom_locations
        }])

        # Predict monthly sales
        prediction = model.predict(input_data)

        # Convert prediction into a non-negative whole number
        result = max(0, round(float(prediction[0])))

        # Display prediction result
        return render_template('result.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)