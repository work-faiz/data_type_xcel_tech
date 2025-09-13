# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /code

# 3. Install Tesseract-OCR and other system dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 5. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the application code into the container
COPY ./main.py /code/main.py

# 7. Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]