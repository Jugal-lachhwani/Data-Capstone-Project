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
from src.logger import logging
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import dagshub

nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
mlflow.set_tracking_uri("https://dagshub.com/Jugal-lachhwani/Data-Capstone-Project.mlflow")

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
dagshub.init(repo_owner='Jugal-lachhwani', repo_name='Data-Capstone-Project', mlflow=True)
# Load model once when the app starts
model_name = "my_model"
model_stage = "Staging"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

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

def apply_Tfidf(df):

    X_test = df['review'].values
    
    X_test_bow = vectorizer.transform(X_test)

    test_df = pd.DataFrame(X_test_bow.toarray())
    
    return test_df

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    review = data.text
    text = preprocess_text(review)
    df = apply_Tfidf(pd.DataFrame({'review':[text]}))
    prediction = model.predict(df)
    if prediction == 0:
        return 'Negative'
    else:
        return "Positive"
