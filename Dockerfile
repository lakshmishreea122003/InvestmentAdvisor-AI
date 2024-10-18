# FROM python:3.9-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py .

# EXPOSE 8501

# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]



# Use the official image as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Run the command to start Streamlit
CMD ["streamlit", "run", "app.py"]
