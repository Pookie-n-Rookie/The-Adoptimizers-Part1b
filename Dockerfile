FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY . .

# Run model download script at build-time
RUN python download_model.py

ENTRYPOINT ["python", "main.py"]