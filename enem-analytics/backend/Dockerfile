FROM python:3.11-slim

WORKDIR /app

# Copy all files (expects Root Directory set to enem-analytics/backend in Railway)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
