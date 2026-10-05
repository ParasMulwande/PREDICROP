# 🌱 PREDICROP – Smart Crop Management

PREDICROP is an AI/ML-powered agriculture web application that helps growers decide **what to plant**, **how to feed it**, and **what is wrong with it**. It combines a scikit-learn crop recommendation model, a PyTorch ResNet9 plant disease classifier, a rule-based fertilizer adviser, and live weather data from OpenWeather into a single Flask application. The interesting engineering is not only the models — it is making a ~25 MB deep-learning model run reliably inside a 512 MB cloud memory budget.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)

🚀 **Live Demo:** https://predicrop.onrender.com/
💻 **Source Code:** https://github.com/ParasMulwande/PREDICROP

---

## ✨ Features

### 🌾 Crop Recommendation

A supervised classifier maps soil chemistry and local climate to a crop recommendation.

**Inputs:** nitrogen (N), phosphorus (P), potassium (K), soil pH, and rainfall — plus temperature and humidity retrieved automatically for the selected city. Temperature and humidity are **not** typed by the user; the backend fetches them so the feature vector always reflects real local conditions.

**Output:** one of **22 crop classes**.

The trained estimator is a `RandomForestClassifier` (`n_estimators=20`, `random_state=0`) serialised to `app/models/RandomForest.pkl` and loaded lazily on first request.

### 🧪 Fertilizer Recommendation

A rule-based adviser — deliberately not a model — that answers a narrower question: *which single nutrient is furthest from ideal for this crop?*

**Inputs:** crop name, plus the N, P, and K currently measured in the soil.

**How it works:** `app/Data/fertilizer.csv` holds ideal N/P/K values per crop. The route subtracts the user's readings from those ideals, identifies the nutrient with the largest absolute deviation, and returns guidance from `app/utils/fertilizer.py` — one of six outcomes (`NHigh`, `Nlow`, `PHigh`, `Plow`, `KHigh`, `Klow`), each explaining the likely agronomic effect and practical amendments such as manure, coffee grinds, or nitrogen-fixing legumes.

Keeping this logic explicit rather than learned means the recommendation is inspectable and traceable to a specific number.

### 🍃 Plant Disease Detection

Upload a leaf image and the app classifies it into one of **38 crop–disease classes**, then returns the cause of the disease and prevention guidance from `app/utils/disease.py`.

**Model:** a custom **ResNet9** convolutional network (`app/utils/model.py`, ~6.6 M parameters) implemented in PyTorch, with weights in `app/models/plant_disease_model.pth` (~25 MB). Classes span Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

Because the service runs on a constrained cloud instance, this path is deliberately memory-conscious:

| Technique | Purpose |
|---|---|
| Lazy, cached model loading | Torch is imported only on the first disease request, never at application start, and the loaded model is reused |
| `torch.inference_mode()` | Autograd bookkeeping is disabled, so no activation is retained for a backward pass that never happens |
| Fixed `Resize((256, 256))` | The classifier is `Linear(512, 38)` after a 4×4×4×4 downsample, so input must resolve to exactly 1×1 spatial — 256×256 is the architectural floor |
| `Image.draft()` | JPEG is decoded at reduced DCT scale, making memory proportional to the *output* size instead of the source resolution |
| `MAX_CONTENT_LENGTH` | Uploads above 16 MB are rejected before decoding |

Measured effect on the 512 MB budget: peak working set fell from **606.7 MB → 427.2 MB** for a 48 MP upload, and peak is now flat regardless of source image resolution.

### 🌦️ Weather Forecast

Backed by the [OpenWeather](https://openweathermap.org/api) API through a small server-side proxy at `/api/weather`, so the API key is never sent to the browser.

The proxy geocodes the city name to coordinates, then requests the 3-hour / 5-day forecast endpoint. The page derives current conditions from the first forecast slot — temperature, humidity, and wind speed — and renders a 5-day forecast strip. Users can search for a city, pick a state/city pair from an India-focused dropdown (`app/static/scripts/cities.js`), or use browser geolocation.

Crop recommendation reuses the same current-conditions lookup, which is why a configured API key is required for that feature.

---

## 🧠 Machine Learning

### Models in this repository

| Task | Model | Framework | Artifact | Classes |
|---|---|---|---|---|
| Crop recommendation | `RandomForestClassifier` (n=20) | scikit-learn | `app/models/RandomForest.pkl` | 22 |
| Plant disease detection | Custom ResNet9 | PyTorch | `app/models/plant_disease_model.pth` | 38 |
| Fertilizer advice | Rule-based lookup | — | `app/Data/fertilizer.csv` + `app/utils/fertilizer.py` | 6 outcomes |

### Preprocessing actually implemented

- **Disease inference:** `Resize((256, 256))` → `ToTensor()`. No channel normalisation is applied.
- **Crop inference:** raw numeric features assembled into a NumPy array in fixed order — `N, P, K, temperature, humidity, ph, rainfall`.
- **Fertilizer:** tabular lookup with pandas against a committed CSV.

### Training data

| Dataset | Location | Notes |
|---|---|---|
| Crop recommendation | `Data-processed/crop_recommendation.csv` | 2,200 rows × 7 features, 22 crop labels |
| Fertilizer reference | `Data-processed/fertilizer.csv`, `app/Data/fertilizer.csv` | Ideal N/P/K per crop |
| Disease images | PlantVillage-derived (see notebooks) | Not committed; only the trained weights are |

Exploratory analysis and model comparison live in `notebooks/` (gitignored), including a RandomForest accuracy check on a 20% holdout (`test_size=0.2`, `random_state=2`) — a sanity check on the training split, **not** a field evaluation.

### A note on model compatibility

The committed `RandomForest.pkl` was originally serialised by scikit-learn 0.23.2. scikit-learn 1.3 added an eighth field (`missing_go_to_left`) to its internal tree-node struct and now rejects older arrays strictly:

```
ValueError: node array from the pickle has an incompatible dtype
```

Because scikit-learn 1.3.1 is the first release with Python 3.12 wheels, **every** version installable on the deployment runtime refuses the old artifact — pinning back is not an option. The fix was to retrain the estimator under the current release using the hyperparameters from the original notebook, translating the removed `max_features='auto'` default to its modern equivalent `max_features=None` so predictions stay comparable. Details are in [DESIGN.md](DESIGN.md).

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────┐
                         │        Browser          │
                         │  Jinja2 templates +     │
                         │  vanilla JS / CSS       │
                         └────────────┬────────────┘
                                      │  HTTP form posts
                                      ▼
                         ┌─────────────────────────┐
                         │   Flask Application     │
                         │   (app/app.py)          │
                         │  ┌───────────────────┐  │
                         │  │  Route handlers   │  │
                         │  └─────────┬─────────┘  │
                         └────────────┼────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             ▼                        ▼                        ▼
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  ML inference    │     │  Rule lookup     │     │  Weather proxy   │
   │  sklearn / torch │     │  pandas + CSV    │     │  OpenWeather API │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            │                        │                        │
   ┌────────▼─────────┐     ┌────────▼─────────┐     ┌────────▼─────────┐
   │ models/*.pkl     │     │ Data/fertilizer  │     │ /api/weather     │
   │ models/*.pth     │     │ utils/*.py       │     │ (key stays       │
   │ (committed)      │     │                  │     │  server-side)    │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            └────────────────────────┼────────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │  Result page rendered   │
                         │  back to the user       │
                         └─────────────────────────┘
```

### Request lifecycle notes

- **Model loading is lazy and idempotent.** Both models load on first use behind a lock, then cache — including caching a *failure*, so a broken environment is attempted only once rather than on every request.
- **Torch never loads at import time.** If PyTorch cannot initialise, the rest of the application keeps serving and only the disease route degrades.
- **Weather failures fail closed.** With no API key configured, crop recommendation returns an explanatory page instead of guessing.

---

## 🛠️ Technology Stack

### Frontend

| Technology | Role |
|---|---|
| Jinja2 templates | Server-rendered pages (`app/templates/`) |
| Vanilla JavaScript | Weather dashboard, fetch calls, geolocation |
| CSS (`css2/predicrop-v2.css`) | Custom design system for the app shell |
| Bootstrap 4 CSS | Grid and utility classes |
| Font Awesome | Iconography |

### Backend

| Technology | Role |
|---|---|
| Flask 3.x | Web framework |
| Gunicorn | Production WSGI server |
| Requests | OpenWeather HTTP client |
| Pillow | Image decoding |
| NumPy / pandas | Feature assembly and tabular lookup |

### Machine Learning / Deep Learning

| Technology | Role |
|---|---|
| scikit-learn | RandomForest crop classifier |
| PyTorch | ResNet9 disease classifier |
| torchvision | Image transforms |
| Pickle | Model serialisation |

### APIs

| API | Usage |
|---|---|
| OpenWeather Geocoding | City name → latitude/longitude |
| OpenWeather Forecast | 3-hour / 5-day forecast data |

### Storage

No database is used. The application is stateless: reference data lives in committed CSV files and learned parameters in committed model binaries, both read-only at runtime. The only runtime state is the in-process model cache.

### Deployment

| Technology | Role |
|---|---|
| Render Web Service | Hosting |
| Python 3.12.8 | Runtime (`app/Runtime.txt`) |
| Gunicorn, 1 worker | Process model (see below) |

### Development tools

Git, virtual environments (`venv`), Jupyter (notebooks, gitignored), `gh` CLI.

---

## 📁 Project Structure

```
PREDICROP/
├── app/                          # Flask application — the deployable root
│   ├── app.py                    # Routes, model loading, inference
│   ├── config.py                 # Reads OPENWEATHER_KEY from the environment
│   ├── config.example.py         # Secret-free config template
│   ├── requirements.txt          # Authoritative dependency list
│   ├── Procfile                  # Gunicorn start command
│   ├── Runtime.txt               # python-3.12.8
│   ├── Data/
│   │   └── fertilizer.csv        # Ideal N/P/K per crop
│   ├── models/                   # Committed — required at runtime
│   │   ├── RandomForest.pkl      #   Crop classifier
│   │   └── plant_disease_model.pth  # Disease classifier (ResNet9)
│   ├── static/
│   │   ├── css/  css2/           # Stylesheets
│   │   ├── images/               # Hero and feature imagery
│   │   └── scripts/cities.js     # India state → city mapping
│   ├── templates/                # Jinja2 pages
│   └── utils/
│       ├── model.py              # ResNet9 architecture
│       ├── disease.py            # 38 class → guidance text
│       └── fertilizer.py         # 6 outcome → guidance text
│
├── Data-processed/               # Training CSVs (committed)
│   ├── crop_recommendation.csv
│   └── fertilizer.csv
│
├── DEPLOY.md                     # Hosting instructions
├── DESIGN.md                     # Architecture & design notes
├── CONTRIBUTING.md
├── LICENSE                       # GPL-3.0
└── requirements.txt              # ⚠️ Stale snapshot, not used for deploys
```

**Notes**

- The deployable root is **`app/`**. Render is configured with `app` as its root directory, so `app/requirements.txt` is the authoritative dependency list.
- **`app/models/` must stay committed.** The application cannot predict without those two files.
- The root `requirements.txt` is a leftover full-environment snapshot that pins long-superseded versions (e.g. Flask 1.1.2). It is **not** used for installation or deployment.
- Local-only workspace directories (`notebooks/`, `Data-raw/`, root `models/`) are gitignored to avoid duplicating large binaries in the repository.

---

## ⚙️ Local Installation

PREDICROP runs from the `app/` directory.

**1. Clone the repository**

```bash
git clone https://github.com/ParasMulwande/PREDICROP.git
cd PREDICROP/app
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are pulled from the official CPU wheel index declared at the top of the file, so no CUDA download is required.

> **Tip:** Python **3.12** matches the deployment runtime and is the best-tested version. Newer interpreters can fail to resolve native wheels for the compiled scientific stack.

**4. Configure your OpenWeather API key**

Get a key from [OpenWeatherMap](https://openweathermap.org/api), then either set an environment variable:

```bash
# Windows (PowerShell)
$env:OPENWEATHER_KEY="your_api_key_here"

# macOS / Linux
export OPENWEATHER_KEY="your_api_key_here"
```

…or create a local, gitignored config file:

```bash
copy config.example.py config_local.py     # Windows
# cp config.example.py config_local.py     # macOS / Linux
```

```python
# config_local.py
weather_api_key = "your_api_key_here"
```

`config_local.py` is listed in `.gitignore`, so your key stays on your machine. The environment variable takes precedence, and that is the path used in production.

**5. Run the application**

```bash
python app.py
```

**6. Open the app**

<http://127.0.0.1:5000>

Without an API key the application still starts; the weather page and crop recommendation fail closed with an explanatory message rather than erroring.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENWEATHER_KEY` | For weather and crop recommendation | OpenWeatherMap API key. Read in `app/config.py`. |

```bash
OPENWEATHER_KEY=your_api_key_here
```

> **Never commit API keys or secrets.** `config_local.py`, `.env`, `.env.*`, `*.key`, and `*.pem` are all excluded by `.gitignore`. `config.py` treats unconfigured placeholder values as a missing key so the app fails with a clear message instead of a confusing upstream error. If a key is ever committed, treat it as compromised and rotate it at the provider.

---

## 🚀 Deployment

PREDICROP is deployed as a **Render Web Service**.

| Setting | Value |
|---|---|
| Root directory | `app` |
| Build command | `pip install -r requirements.txt` |
| Start command | From `app/Procfile` |
| Runtime | Python 3.12.8 (`app/Runtime.txt`) |
| Health check path | `/` |
| Environment | `OPENWEATHER_KEY` (set as a secret) |

**Procfile**

```
web: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --access-logfile - app:app
```

### Why a single worker

This is a deliberate memory decision, not a default. Each Gunicorn worker is a separate process that independently imports PyTorch (~282 MB measured) and loads the 25 MB disease weights, putting a worker at roughly **324 MB steady state**. Two workers total ~648 MB before serving a single request, which exceeds Render's Free-tier **512 MB** limit regardless of anything the inference path does.

With the preprocessing work described above, one worker peaks at ~427 MB, leaving workable headroom. The trade-off is that disease prediction is served serially — acceptable because the model loads lazily and only the first request is slow.

Other Render properties worth knowing: free instances sleep when idle, so the first request after a quiet period is slow. There is no autoscaling or zero-downtime configuration in this setup; a cold start is expected. See [DEPLOY.md](DEPLOY.md) for the full procedure, including Railway and Heroku notes.

---

## 🧪 Testing / Validation

The following flows were exercised against a local instance running the production dependency set (scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6, PyTorch 2.14.1+cpu) and confirmed on the deployed service:

| Flow | Result |
|---|---|
| Crop recommendation | `POST /crop-predict` → HTTP 200, returns a crop class |
| Fertilizer recommendation | `POST /fertilizer-predict` → HTTP 200, returns nutrient guidance |
| Disease detection | `POST /disease-predict` → HTTP 200 with an image upload, returns a valid class from the 38-label set |
| Weather | `GET /api/weather?city=…` → HTTP 200, returns a populated forecast |
| Page routes | `/`, `/crop-recommend`, `/fertilizer`, `/disease-predict`, `/weather`, `/addinfo` → HTTP 200 |

**Compatibility and parity checks**

- The disease model's numeric output under `inference_mode()` was confirmed **bit-identical** to the grad-tracked path via `torch.equal`, so the memory optimisation provably cannot alter predictions.
- The disease transform was verified to produce a `(3, 256, 256)` tensor for every input shape tested, across JPEG, PNG, WEBP, and BMP.
- The retrained crop classifier was confirmed to load and predict under the deployment's scikit-learn version with no warnings, and its estimator contract (feature count, class count, hyperparameters) matches the artifact it replaced.

This project has no automated test suite; the checks above were performed manually. Adding one is listed under Future Enhancements.

---

## 📸 Screenshots

> **Screenshots have not yet been added to this repository.** Capture the pages below from the live demo and commit them here (e.g. under `docs/`), then replace these placeholders with the real image paths.

| Screen | Live |
|---|---|
| Home | https://predicrop.onrender.com/ |
| Crop recommendation | https://predicrop.onrender.com/crop-recommend |
| Fertilizer recommendation | https://predicrop.onrender.com/fertilizer |
| Disease detection | https://predicrop.onrender.com/disease-predict |
| Weather dashboard | https://predicrop.onrender.com/weather |

Existing committed imagery that is available today:

<p align="center">
  <img src="app/static/images/predicrop-logo.png" alt="PREDICROP logo" width="180">
</p>

---

## 🔒 Security

- **Secrets are never committed.** The API key is read from the `OPENWEATHER_KEY` environment variable, with an optional gitignored `config_local.py` for local development. `.gitignore` excludes `config_local.py`, `.env`, `.env.*`, `*.key`, and `*.pem`.
- **The API key never reaches the browser.** Weather is fetched server-side and exposed only through `/api/weather`, so the key cannot be read from client-side source or devtools.
- **Uploads are bounded.** `MAX_CONTENT_LENGTH` (16 MB) rejects oversized requests before decoding, and images are reduced to a fixed 256×256 before inference.
- **Resource use is bounded.** Models load lazily behind a lock, a failed load is cached rather than retried per request, and inference runs in `inference_mode()` — so a malformed upload cannot cascade into repeated expensive work.
- **Failure modes fail closed.** A missing API key or an unreachable weather service produces an explanatory page rather than a silent bad prediction.

---

## 🚧 Future Enhancements

These are ideas, **not** implemented functionality:

- **Automated test suite** — unit tests for the model contract, transform shapes, and route behaviour, so regressions are caught without manual verification.
- **Additional datasets and crops** — broader regional crop coverage beyond the current 22 classes and India-focused city list.
- **Stronger disease validation** — evaluate the classifier on held-out field imagery rather than the training distribution, and consider a larger or augmented leaf dataset.
- **Model improvements** — explore modern architectures or transfer learning for the disease classifier, and tune the crop model now that retraining is a supported workflow.
- **More agronomic insight** — richer soil and yield modelling beyond a single nutrient recommendation.
- **Historical weather analysis** — trends and seasonal summaries rather than only current conditions and a 5-day outlook.
- **Upload safeguards for non-JPEG formats** — `Image.draft()` optimises JPEG decoding; bounding decoded pixel count for PNG uploads is not yet implemented.
- **Progressive web app** — offline-friendly shell and installable mobile experience.

---

## 👨‍💻 Author

**Paras Mulwande**

- GitHub: https://github.com/ParasMulwande

---

## 📄 License

Released under the **GNU General Public License v3.0**. See [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

PREDICROP is an educational and demonstration project. The datasets used here are compiled from public sources and carry no guarantee of agronomic validity. **Do not use the output of this application as the sole basis for a farming decision.** The intent is to demonstrate how machine learning and deep learning can be applied to precision agriculture.

## 🙏 Credits

This project builds on the open-source **Harvestify** project by [Gladiator07](https://github.com/Gladiator07/Harvestify), which provided the original crop and fertilizer recommendation implementations and the base Flask structure. PREDICROP extends it with a revised interface, a modernised dependency and deployment setup, a re-serialised crop model compatible with current scikit-learn, and substantial work on memory-safe deep-learning inference for constrained cloud hosting.

Datasets:

- [Crop recommendation dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
- [New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)