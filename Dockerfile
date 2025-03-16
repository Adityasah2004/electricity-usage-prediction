# Use an official Python runtime as a base image
FROM python:3.10-slim
FROM tensorflow/tensorflow:latest-gpu


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Expose any ports if needed (e.g., 5000)
EXPOSE 5000

# Command to run your main script (adjust as needed)
CMD ["python", "src/models/train.py"]