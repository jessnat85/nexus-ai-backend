FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    pip install --upgrade pip

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

# Load environment variables from file
ARG OPENAI_API_KEY
ARG NEWSDATA_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV NEWSDATA_API_KEY=${NEWSDATA_API_KEY}

EXPOSE 10000
CMD ["uvicorn", "chart_ai_backend:app", "--host", "0.0.0.0", "--port", "10000"]