FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app/backend
ENV PORT=8080

EXPOSE 8080 8501