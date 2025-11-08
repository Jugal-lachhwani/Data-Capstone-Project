from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import mlflow
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
import os
import mlflow
import pickle

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import Request
from fastapi.logger import logger
from fastapi import status

load_dotenv()

# Set MLflow tracking URI and credentials from environment variables
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Set authentication using environment variables (if provided)
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Vectorizer will be loaded from MLflow run artifacts at startup.
# Initialize as None so startup logic downloads and loads it from the registry.
vectorizer = None

# Configure model name and stage via environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "my_model")
# Default to Production as requested; can be overridden with MODEL_STAGE env var
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")

# Placeholder for loaded model and status
model = None
model_loaded = False

def load_model_from_registry(name: str, stage: str):
    """Try to load model from MLflow registry for given name and stage."""
    uri = f"models:/{name}/{stage}"
    logger.info('Attempting to load model from %s', uri)
    return mlflow.pyfunc.load_model(model_uri=uri)

def download_vectorizer_from_registry(name: str, stage: str, dst_dir: str):
    """Download vectorizer artifact from the run that produced the specified model stage.
    Returns local path to downloaded vectorizer or None if not found.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        # Find latest model version for the stage
        versions = client.search_model_versions(f"name='{name}'")
        target = None
        for v in versions:
            if v.current_stage.lower() == stage.lower():
                target = v
                break
        if target is None:
            logger.warning('No model version found for %s stage %s', name, stage)
            return None
        run_id = target.run_id
        # Try known artifact locations where register_model.py uploads the vectorizer
        candidates = [
            'model_artifacts/vectorizer.pkl',
            'vectorizer.pkl',
            'model/vectorizer.pkl'
        ]
        for cand in candidates:
            try:
                local_path = client.download_artifacts(run_id=run_id, path=cand, dst_path=dst_dir)
                if local_path and os.path.exists(local_path):
                    logger.info('Downloaded vectorizer artifact from %s (run %s) to %s', cand, run_id, local_path)
                    return local_path
            except Exception:
                continue
        logger.warning('Vectorizer artifact not found in run %s for candidates %s', run_id, candidates)
        return None
    except Exception as e:
        logger.error('Error while downloading vectorizer from registry: %s', e)
        return None

def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

def apply_Vec(df):

    X_test = df['review'].values
    
    X_test_bow = vectorizer.transform(X_test)

    test_df = pd.DataFrame(X_test_bow.toarray())
    
    return test_df

app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Load model from MLflow registry on startup. Try Production first, fallback to Staging."""
    global model, model_loaded
    try:
        try:
            model = load_model_from_registry(MODEL_NAME, MODEL_STAGE)
            model_loaded = True
            logger.info('Model %s loaded from stage %s', MODEL_NAME, MODEL_STAGE)
        except Exception as e:
            logger.warning('Failed to load model from %s: %s', MODEL_STAGE, e)
            # Fallback: try Staging
            if MODEL_STAGE.lower() != 'staging':
                try:
                    model = load_model_from_registry(MODEL_NAME, 'Staging')
                    model_loaded = True
                    logger.info('Model %s loaded from stage Staging', MODEL_NAME)
                except Exception as e2:
                    logger.error('Failed to load model from Staging: %s', e2)
                    model_loaded = False
            else:
                model_loaded = False
    except Exception as e:
        logger.error('Unexpected error during model loading: %s', e)
        model_loaded = False
    # If vectorizer isn't present locally, try to download it from the model run artifacts
    global vectorizer
    if vectorizer is None and model_loaded:
        models_dir_local = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir_local, exist_ok=True)
        vec_local = download_vectorizer_from_registry(MODEL_NAME, MODEL_STAGE, models_dir_local)
        if vec_local:
            try:
                with open(vec_local, 'rb') as vf:
                    vectorizer = pickle.load(vf)
                logger.info('Vectorizer loaded from registry artifact: %s', vec_local)
            except Exception as e:
                logger.error('Failed to load downloaded vectorizer: %s', e)

class InputData(BaseModel):
    text: str
    
@app.get("/")
def predict():
    return {"Hello": "World"}


@app.get('/health')
def health():
    """Return health status including whether model was loaded."""
    return {"model_loaded": model_loaded}

@app.post("/predict")
def predict(data: InputData):
    if not model_loaded:
        # Fail fast with 503: service not ready
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    review = data.text
    text = preprocess_text(review)
    if vectorizer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vectorizer not available")

    df = apply_Vec(pd.DataFrame({'review':[text]}))
    try:
        prediction = model.predict(df)
    except Exception as e:
        logger.error('Prediction failed: %s', e)
        raise HTTPException(status_code=500, detail=str(e))

    # prediction may be array-like
    pred_val = prediction[0] if hasattr(prediction, '__iter__') else prediction
    if int(pred_val) == 0:
        return 'Negative'
    else:
        return 'Positive'


