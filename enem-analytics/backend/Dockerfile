FROM python:3.11-slim

WORKDIR /app

# Copy backend files (context is repo root)
COPY enem-analytics/backend/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
