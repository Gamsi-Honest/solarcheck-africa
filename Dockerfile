# SolarCheck Africa — GSI mode
#
# Container image for permanent hosting (Render, Fly.io, Railway, Cloud Run,
# any Docker host). For local or preview use you do not need this file at all:
#   streamlit run app.py
#
# Build:  docker build -t solarcheck-gsi .
# Run:    docker run -p 8501:8501 -e GEMINI_API_KEY=... solarcheck-gsi

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first: this layer stays cached across code changes, so redeploys
# only rebuild the app itself.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop root. The app only reads its own files.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit exposes this itself; hosting platforms use it for readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

# Hosts like Render and Railway inject $PORT and expect the app to honour it.
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.fileWatcherType none --server.enableCORS false --server.enableXsrfProtection false"]
