FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    pip install --upgrade pip

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 10000
CMD ["uvicorn", "chart_ai_backend:app", "--host", "0.0.0.0", "--port", "10000"]
