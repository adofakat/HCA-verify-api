FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Copy cookies file if present (needed for TikTok/Instagram)
# The glob pattern cookies.tx[t] means Docker won't fail if the file is missing
COPY cookies.tx[t] /app/cookies.txt

# Run
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
