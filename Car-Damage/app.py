import os
import io
import json
import base64
import numpy as np
import cv2
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_car_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    damage_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)
    total_area = img.shape[0] * img.shape[1]
    damage_ratio = damage_area / total_area

    multiplier = 1.0
    severity = "Undamaged"

    if damage_ratio > 0.05:
        multiplier = 0.5
        severity = "Severe"
    elif damage_ratio > 0.01:
        multiplier = 0.7
        severity = "Moderate"
    elif damage_ratio > 0.001:
        multiplier = 0.9
        severity = "Minor"
    else:
        multiplier = 1.0
        severity = "Negligible"

    highlighted_img = img.copy()
    cv2.drawContours(highlighted_img, contours, -1, (0, 0, 255), 3)

    _, buffer = cv2.imencode('.jpg', highlighted_img)
    highlighted_img_b64 = base64.b64encode(buffer).decode('utf-8')

    return multiplier, severity, highlighted_img_b64

def fetch_market_prices(make, model, year, mileage):
    search_url = f"http://example-car-marketplace.com/search?make={make}&model={model}&year={year}"
    prices = []
    similar_cars = []

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        listings = soup.find_all('div', class_='listing-card')

        for listing in listings[:5]:
            price_tag = listing.find('span', class_='price-value')
            details_tag = listing.find('p', class_='car-details')

            if price_tag and details_tag:
                try:
                    price_str = price_tag.text.strip().replace('$', '').replace(',', '')
                    price = float(price_str)
                    prices.append(price)

                    car_info = details_tag.text.strip()
                    similar_cars.append({"details": car_info, "price": price})
                except ValueError:
                    continue

    except Exception as e:
        print(f"Scraping failed: {e}")

    return prices, similar_cars

def calculate_final_value(prices, damage_multiplier):
    if not prices:
        return 0.0, 0.0

    average_price = sum(prices) / len(prices)
    estimated_value = average_price * damage_multiplier

    return average_price, estimated_value

@app.route('/api/valuate', methods=['POST'])
def valuate_car():
    if 'car_image' not in request.files:
        return jsonify({"error": "No car image provided"}), 400

    file = request.files['car_image']
    image_bytes = file.read()

    make = request.form.get('make')
    model = request.form.get('model')
    year = request.form.get('year')
    mileage = request.form.get('mileage')

    if not all([make, model, year]):
        return jsonify({"error": "Missing essential car details (make, model, year)"}), 400

    try:
        multiplier, severity, highlighted_img_b64 = process_car_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Computer Vision processing failed: {e}"}), 500

    prices, similar_cars = fetch_market_prices(make, model, year, mileage)

    if not prices:
        return jsonify({"error": "Could not fetch current market prices. Check scraping logic or site availability."}), 500

    average_price, final_value = calculate_final_value(prices, multiplier)

    response_data = {
        "status": "success",
        "input_details": {"make": make, "model": model, "year": year},
        "damage_severity": severity,
        "damage_multiplier": multiplier,
        "highlighted_image_b64": highlighted_img_b64,
        "fetched_car_prices": prices,
        "average_market_price": round(average_price, 2),
        "final_estimated_value": round(final_value, 2),
        "similar_cars_table": similar_cars
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)