FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY code/deployment/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY code/deployment/app /app/code/deployment/app

EXPOSE 8501

ENV API_URL=http://api:8000

CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

