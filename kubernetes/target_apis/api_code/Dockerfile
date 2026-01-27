FROM python:3.12.12

WORKDIR /app

# Install FFmpeg for video encoding
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -r req.txt
EXPOSE 8000

CMD ["python", "server.py"]
