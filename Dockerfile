# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip and install required packages
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit command when the container launches
CMD ["streamlit", "run", "app.py"]
