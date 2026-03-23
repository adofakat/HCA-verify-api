FROM python:3.11-slim

# Install ffmpeg + Node.js (required by yt-dlp for TikTok JS challenge solving)
RUN apt-get update && \
    apt-get install -y ffmpeg nodejs npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Copy cookies file if present (needed for TikTok/Instagram downloads)
COPY cookies.tx[t] /app/cookies.txt

# Run
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
