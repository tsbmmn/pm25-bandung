# Satu image untuk API (default) atau UI (sama seperti template P3).
#   docker build -t pm25-bandung .
#   docker run -p 7860:7860 pm25-bandung              -> FastAPI di http://localhost:7860/docs
#   docker run -p 7860:7860 -e APP=ui pm25-bandung    -> Streamlit di http://localhost:7860
FROM python:3.14-slim

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY artifacts ./artifacts

ENV PORT=7860 APP=api
EXPOSE 7860
RUN useradd -m appuser
USER appuser
CMD if [ "$APP" = "ui" ]; then \
      streamlit run app/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0; \
    else \
      uvicorn app.api.main:app --host 0.0.0.0 --port $PORT; \
    fi
