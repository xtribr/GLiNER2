FROM python:3.11-slim

WORKDIR /app

# Copy backend files from enem-analytics/backend
COPY enem-analytics/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY enem-analytics/backend/api/ ./api/
COPY enem-analytics/backend/ml/ ./ml/
COPY enem-analytics/backend/data/ ./data/
COPY enem-analytics/backend/scripts/ ./scripts/
COPY enem-analytics/backend/database/ ./database/

# Expose port
EXPOSE 8000

# Run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
