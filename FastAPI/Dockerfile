FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scikit-learn & MLflow
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY FastAPI/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY FastAPI/ .

# Copy model files from parent models directory
COPY models/ /app/models/

# Pre-download NLTK data
RUN python -m nltk.downloader wordnet stopwords -d /usr/local/share/nltk_data

# Expose port (match Uvicorn port)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
