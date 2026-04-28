FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy==1.26.4"

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
