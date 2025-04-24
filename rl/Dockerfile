# Use an official PyTorch image as the base
FROM pytorch/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y git


# Set working directory
WORKDIR /app


# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install wandb

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1


