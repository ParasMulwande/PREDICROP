# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import logging
import threading
from PIL import Image

logger = logging.getLogger('predicrop')


class DiseaseModelUnavailable(RuntimeError):
    """Raised when the torch-backed leaf classifier cannot be loaded."""

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'

# The classifier is loaded on first use rather than at import time. Torch ships
# unsigned native wheels, and hosts running Smart App Control / WDAC refuse to
# load them (WinError 4551). Importing torch at module scope aborts this whole
# file and takes down every unrelated route, because the failure happens before
# the app object exists. The failure is cached so it is only attempted once.
_disease_model = None
_disease_model_error = None
_disease_model_lock = threading.Lock()


def get_disease_model():
    """
    Import torch and load the ResNet9 weights, once, on first use.
    :return: the eval-mode model, or None when torch cannot be loaded
    """
    global _disease_model, _disease_model_error

    if _disease_model is not None or _disease_model_error is not None:
        return _disease_model

    with _disease_model_lock:
        if _disease_model is not None or _disease_model_error is not None:
            return _disease_model

        try:
            import torch
            from utils.model import ResNet9

            model = ResNet9(3, len(disease_classes))
            model.load_state_dict(torch.load(
                disease_model_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as error:
            logger.warning(
                'Disease classifier unavailable, '
                '/disease-predict will not classify leaves: %s',
                error)
            _disease_model_error = str(error)
            return None

        _disease_model = model
        logger.info('Disease classifier loaded from %s', disease_model_path)
        return _disease_model


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
_crop_model = None


def get_crop_model():
    global _crop_model
    if _crop_model is None:
        import pickle
        _crop_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
    return _crop_model


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and return the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x.get("cod") != "404" and "main" in x:
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=None):
    """
    Transforms image to tensor and predicts disease label
    :params: img
    :return: prediction (string)
    """
    if model is None:
        model = get_disease_model()

    if model is None:
        raise DiseaseModelUnavailable(
            _disease_model_error or 'PyTorch could not be loaded.')

    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    with torch.inference_mode():
        yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================

# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.static_folder = 'static'

# The dev server runs with debug off, which makes Jinja cache compiled templates
# for the life of the process. Without this, template edits are served stale
# until the server is restarted.
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# render home page


@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


@app.route('/weather')
def weather():
    title = 'Predicrop - Weather Forecasting'
    return render_template(
        'weather.html',
        weather_api_key=config.weather_api_key
    )


@app.route('/api/weather')
def weather_api():

    city = request.args.get('city', '').strip()

    if not city:
        return {"error": "Please enter a city name."}, 400

    try:
        api_key = config.weather_api_key

        if not api_key:
            return {
                "error": "Weather API key is not configured."
            }, 500

        # --------------------------------
        # STEP 1: Find city coordinates
        # --------------------------------
        geo_url = "https://api.openweathermap.org/geo/1.0/direct"

        geo_params = {
            "q": city,
            "limit": 1,
            "appid": api_key
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_data = geo_response.json()

        # Return the actual OpenWeather error
        if geo_response.status_code != 200:
            return {
                "error": geo_data.get(
                    "message",
                    "Unable to find city."
                )
            }, geo_response.status_code

        if not geo_data:
            return {
                "error": f"City '{city}' was not found."
            }, 404

        latitude = geo_data[0]["lat"]
        longitude = geo_data[0]["lon"]
        city_name = geo_data[0]["name"]

        # --------------------------------
        # STEP 2: Get weather forecast
        # --------------------------------
        forecast_url = (
            "https://api.openweathermap.org/data/2.5/forecast"
        )

        forecast_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key
        }

        forecast_response = requests.get(
            forecast_url,
            params=forecast_params,
            timeout=10
        )

        forecast_data = forecast_response.json()

        if forecast_response.status_code != 200:
            return {
                "error": forecast_data.get(
                    "message",
                    "Unable to fetch weather forecast."
                )
            }, forecast_response.status_code

        return {
            "city": city_name,
            "forecast": forecast_data.get("list", [])
        }

    except requests.RequestException:
        return {
            "error": "Unable to connect to the weather service."
        }, 503

    except Exception as error:
        return {
            "error": str(error)
        }, 500


@app.route('/addinfo')
def addinfo():
    title = 'Predicrop - More Information'
    return render_template('more-info.html')


# render crop recommendation form page

@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        city = request.form.get("city")

        weather_res = weather_fetch(city) if city else None
        if weather_res is not None:
            temperature, humidity = weather_res
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = get_crop_model().predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    import pandas as pd
    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title, crop_name=crop_name)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    classifier_ready = get_disease_model() is not None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('disease.html', title=title,
                                   model_unavailable=not classifier_ready)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except DiseaseModelUnavailable as error:
            logger.warning('Disease request refused: %s', error)
        except Exception:
            logger.exception(
                'Disease prediction failed for the uploaded file')

    return render_template('disease.html', title=title,
                           model_unavailable=not classifier_ready)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)