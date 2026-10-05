"""Application configuration.

The OpenWeather API key is read from the OPENWEATHER_KEY environment variable so
that no secret is ever committed to version control.

Local development: create an untracked config_local.py beside this file with

    weather_api_key = "your-key-here"

config_local.py is gitignored, so the key stays on your machine only.

Hosting (Render, Railway, Heroku, ...): set OPENWEATHER_KEY in the service's
environment variables instead, and do not create a config_local.py.

If no key is configured the weather routes fail closed with a clear message
rather than crashing.
"""
import os

# Values that mean "not actually configured". Treating them as empty avoids a
# confusing 401 from the weather API when a placeholder is left in place.
_PLACEHOLDERS = {
    "",
    "your-key-here",
    "paste-your-openweathermap-key-here",
    "your new rotated openweather api key",
    "your_openweather_key",
}

weather_api_key = os.environ.get("OPENWEATHER_KEY", "").strip()

if not weather_api_key:
    try:
        from config_local import weather_api_key as _local_key
    except ImportError:
        pass
    else:
        weather_api_key = (_local_key or "").strip()

if weather_api_key.strip("<>").strip().lower() in _PLACEHOLDERS:
    weather_api_key = ""