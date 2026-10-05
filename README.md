# 🌱 PREDICROP – Smart Crop Management

PREDICROP is an AI-powered smart agriculture web application designed to assist farmers and agricultural users with data-driven crop and plant health insights.

The application combines Machine Learning, Deep Learning, and weather data to provide crop recommendations, fertilizer guidance, plant disease detection, and weather information through a single web platform.

🚀 **Live Demo:** https://predicrop.onrender.com/

---

## ✨ Features

### 🌾 Crop Recommendation

PREDICROP recommends a suitable crop based on agricultural and environmental inputs such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH
- Rainfall
- Temperature
- Humidity

The recommendation is generated using a trained Machine Learning model.

---

### 🧪 Fertilizer Recommendation

The fertilizer module provides fertilizer guidance based on the selected crop and available soil/nutrient information.

It is designed to help users understand suitable fertilizer recommendations for their agricultural conditions.

---

### 🍃 Plant Disease Detection

Users can upload a plant leaf image and PREDICROP analyzes the image using a Deep Learning-based disease classification model.

The system provides:

- Predicted crop
- Detected disease
- Disease information
- Cause/background information
- Prevention and treatment guidance

The deployed application uses memory-conscious image preprocessing to support Deep Learning inference on a resource-constrained cloud environment.

---

### 🌦️ Weather Information

PREDICROP integrates weather data to provide location-based weather information.

The weather module uses the **OpenWeather API** to retrieve current weather information for the selected location.

---

## 🧠 Machine Learning

PREDICROP combines traditional Machine Learning and Deep Learning techniques.

### Crop Recommendation
- Random Forest
- Scikit-learn
- Agricultural/environmental input features

### Plant Disease Detection
- ResNet9-based Deep Learning model
- PyTorch
- Image preprocessing and classification

The models are integrated directly into the Flask application to provide predictions through the web interface.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PREDICROP Web UI  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Flask Backend    │
                    └───────┬─────┬───────┘
                            │     │
              ┌─────────────┘     └──────────────┐
              ▼                                   ▼
    ┌──────────────────┐                ┌──────────────────┐
    │   ML Models      │                │  Weather API     │
    │                  │                │                  │
    │ Crop Prediction  │                │    OpenWeather   │
    │ Disease Detection│                │                  │
    └────────┬─────────┘                └────────┬─────────┘
             │                                   │
             └────────────────┬──────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Results & Insights  │
                    └─────────────────────┘
