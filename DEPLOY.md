# Deploying PREDICROP

The app is a standard Flask/gunicorn service, so it runs on any host that can
build Python and start a `Procfile`.

Two things matter before it works in production:

1. **`OPENWEATHER_KEY` must be set** as an environment variable. The key is no
   longer stored in the repository.
2. **`app/models/` must be present.** Those artifacts are committed, so a normal
   clone can predict. Do not delete them.

---

## Render (recommended)

Render's free Python web service is the least fiddly option.

1. Push this repository to GitHub.
2. In Render: **New → Web Service → connect your repository.**
3. Fill in the service settings:

   | Setting | Value |
   |---|---|
   | Root Directory | `app` |
   | Runtime | Python 3.12 (or newer) |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app` |
   | Health Check Path | `/` |

   Leaving **Start Command** empty also works — Render reads `app/Procfile`,
   which contains exactly that gunicorn line.

4. Under **Environment**, add:
   - `OPENWEATHER_KEY` → your OpenWeatherMap API key

   Do **not** create a `config_local.py` in production; the environment variable
   is the supported path.

5. Click **Create Web Service**, then watch the deploy log.

### Notes and limits on the free tier

- The torch model (~25 MB) is loaded **lazily on the first image upload**, which
  can take several seconds. The `--timeout 120` in the Procfile covers this; if
  you see 502s on the first disease prediction, raise it.
- Free instances sleep when idle, so the first request after a quiet period is
  slow. Crop and fertilizer prediction is cheap; disease detection is the slow
  path.
- `scikit-learn` emits a `UserWarning` because the pickles were serialised with
  0.23.2 and the runtime is newer. Predictions work, but retrain the artifacts
  to remove the warning. See DESIGN.md §9.

---

## Railway

1. **New Project → Deploy from GitHub repo.**
2. Railway detects `app/Procfile` automatically. If it does not, set the root
   directory to `app` and the start command to the gunicorn line above.
3. Add the `OPENWEATHER_KEY` variable under the service's **Variables** tab.

---

## Heroku

`app/Runtime.txt` pins `python-3.12.8` and `app/Procfile` provides the start
command, so this should work as-is. Remember to set `OPENWEATHER_KEY` under
**Settings → Config Vars**.

Note the upstream project deployed from a separate `deploy` branch containing
only the app. This repository keeps one branch with `app/models/` committed, so
no branch juggling is needed.

---

## Verifying a deploy

Once the service is up:

```bash
curl -o /dev/null -s -w "%{http_code}\n" https://<your-host>/           # 200
curl -o /dev/null -s -w "%{http_code}\n" https://<your-host>/crop-recommend
curl "https://<your-host>/api/weather?city=Nagpur"                    # JSON, proves the key works
```

If `/api/weather` returns a 500 with "Weather API key is not configured", the
`OPENWEATHER_KEY` variable is missing from the service environment.

---

## Rotating the API key

The key that previously lived in `app/config.py` should be treated as
compromised and rotated at <https://openweathermap.org/api>. Set the new value
as `OPENWEATHER_KEY` in the host and in your local `config_local.py`.