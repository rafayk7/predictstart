from flask import Flask, jsonify, request, make_response, abort
from flask_cors import CORS

import sys
from scraper import KickstarterScraper

sys.path.insert(0, './datascience/')
from predict import Predictor

app = Flask(__name__)
CORS(app)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route("/predict", methods=['POST'])
def send_prediction():
    if not request.json or not 'url' in request.json:
        abort(400)

    succeeded = False

    url = request.json['url']

    try:
        json = KickstarterScraper(url).scrape(display=True, json=True)
        json["error"] = False
    except:
        json = {"Error": "Error Scraping Data"}
        json['pledged'] = 0
        json['goal'] = 0

    if json['pledged'] > json['goal']:
        json['succeeded'] = True
        succeeded = True
        json["error"] = False
    else:
        json['succeeded'] = False
        
    prediction = Predictor('./datascience/data/finalmodel.pkl')

    if not succeeded:
        try:
            label, acc, imp_vals = prediction.predict(json)
            json["error"] = False
        except:
            label = 2
            acc = True
            imp_vals = {}
            json["error"] = True
            json["ErrorMsg"]  = "Error Obtaining Prediction"

        json['label'] = label
        json['acc'] = acc
        json['imp_vals'] = [imp_vals]

    return make_response(jsonify(json), 201)

    # Scrape URL
    # Get appropriate label
    # Get Value for each feature
    # Send response


if __name__ == '__main__':
    app.run(port=3002, debug=True)
