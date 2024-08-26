# Use the official Python image from DockerHub
FROM python:3.12

# Set the working directory in the Docker image
WORKDIR /backend

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY ./app /backend/app
COPY ./run_api.py /backend/run_api.py

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the app using uvicorn
CMD ["python", "run_api.py"]
