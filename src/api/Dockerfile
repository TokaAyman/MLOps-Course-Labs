# Use the official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy all files to /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=5 --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
