FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY app/ ./app/
COPY data/ ./data/

# Set environment
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start the app
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]