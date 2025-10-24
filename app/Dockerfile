FROM python:3.11-slim

WORKDIR /app

# Install Redis
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

COPY . .
# Copy and install requirements
RUN pip install -r requirements.txt

# Set environment
ENV PYTHONPATH=/app/backend
ENV PORT=8080
ENV REDIS_URL=redis://localhost:6379

# Expose port
EXPOSE 8080

# Start Redis in background, then start the app
CMD redis-server --daemonize yes && uvicorn main:app --host 0.0.0.0 --port 8080