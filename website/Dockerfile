# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Set environment variable for Flask
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]
