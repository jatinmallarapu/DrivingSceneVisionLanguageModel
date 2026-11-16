FROM python:3.11-slim
WORKDIR /app
COPY service-requirements.txt .
RUN pip install --no-cache-dir -r service-requirements.txt
COPY service.py .
USER 10001
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
