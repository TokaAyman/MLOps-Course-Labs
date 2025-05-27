FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -i https://pypi.org/simple/ -r requirements.txt


# Install Prometheus FastAPI instrumentation
RUN pip install prometheus-fastapi-instrumentator

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
