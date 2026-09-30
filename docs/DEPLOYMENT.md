# Deploying SolarCheck Africa (GSI mode)

Three routes, cheapest first. All of them run the **same** application — only the
hosting differs.

---

## 0. Live now, in this session (nothing to do)

The app is already running in the workspace and is visible as a **live preview**
next to the conversation. It binds `0.0.0.0:8501`, which is what the preview proxy
forwards.

> This preview lives **only as long as the session**. It is perfect for showing the
> council to someone, not for putting on a market stall. Use route 2 or 3 for that.

---

## 1. Your own machine / laptop

```bash
git clone https://github.com/Gamsi-Honest/solarcheck-africa.git
cd solarcheck-africa
git checkout arena/01a0f152-solarcheck-africa

pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints two URLs:

```
  Local URL: http://localhost:8501      <- this machine
  Network URL: http://192.168.x.x:8501  <- phones on the same Wi-Fi
```

The **Network URL** is the one to share with vendors in the market — they open it
on a phone, no install needed.

Enable sticker photo-scanning by adding a key (optional — manual mode uses the
identical council):

```bash
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml
```

The committed `.streamlit/config.toml` already binds `0.0.0.0:8501`, so no extra
flags are needed. On a headless server:

```bash
streamlit run app.py --server.address 0.0.0.0 \
                     --server.enableCORS false \
                     --server.enableXsrfProtection false
```

---

## 2. Streamlit Community Cloud — free, permanent, best fit

1. Go to **<https://share.streamlit.io>** → *Create app* → *Deploy a public app from GitHub*.
2. Fill in:
   - **Repository:** `Gamsi-Honest/solarcheck-africa`
   - **Branch:** `arena/01a0f152-solarcheck-africa`
   - **Main file path:** `app.py`
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. **Deploy.** Everything the engine needs is committed — `gsi/`,
   `gsi/data/feature_priors.json`, both `.pkl` model files and the training CSV
   used to fit the probability calibration. Nothing has to be uploaded by hand.

You get a permanent public URL you can print as a QR code.

**Note:** `main` still holds the pre-GSI version. Either select the branch above,
or merge the pull request so `main` becomes the deployable default.

---

## 3. Docker / Render / Railway / Fly.io / Cloud Run

A `Dockerfile` and a `render.yaml` blueprint are included.

### Render (free tier, one click)

Connect the repo at <https://dashboard.render.com>, choose **New → Blueprint**, point
it at `render.yaml`, and set `GEMINI_API_KEY` in the dashboard when prompted. The
blueprint uses `/ _stcore/health` as the readiness check. Nothing secret is committed.

### Any Docker host

```bash
docker build -t solarcheck-gsi .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-key solarcheck-gsi
```

The image honours `$PORT` (injected by Render/Railway), runs as a non-root user,
and ships a healthcheck against Streamlit's own `/_stcore/health` endpoint.

---

## Why CORS and XSRF protection are disabled behind proxies

Streamlit rejects requests whose `Origin` does not match its own host. Behind a
reverse proxy, an iframe, or a preview host, that check fails and the app appears
broken. Disabling `enableCORS` and `enableXsrfProtection` is the documented fix,
and it is what the preview here and the Dockerfile both do.

**Security note:** with XSRF protection off, a malicious page could in principle
poke an open Streamlit session. This app holds no accounts, no database and no
user data — every session is stateless and local to the browser — so the exposure
is low. If you later add logins or stored reports, re-enable XSRF protection and
serve the app on its own origin instead of inside an iframe.

---

## Operational notes

| Concern | Detail |
|---|---|
| **Cold start** | ~2 s to fit the probability calibration (5-fold cross-validation) once per session, then cached. Verdicts are well under a second after that. |
| **Memory** | ~350 MB. Fits every free tier listed above. |
| **Stateless** | No database. Nothing is stored server-side; each browser session is independent. |
| **No API key** | The app runs fully without `GEMINI_API_KEY`; only sticker photo-scanning is disabled, and the UI says so rather than failing silently. |
| **Model loading** | If `solarcheck_model_v2.pkl` is missing, the app stops with a clear message instead of producing a verdict. |
| **Batch / registry use** | `python -m gsi.cli --csv panels.csv --out verdicts.csv` runs the same engine headlessly — no UI needed. |
