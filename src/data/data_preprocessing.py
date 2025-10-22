import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
import yaml
nltk.download('wordnet')
nltk.download('stopwords')

def load_config(config_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', config_path)
        return config
    except FileNotFoundError:
        logging.error('File not found: %s', config_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def preprocess_dataframe(df, col='text',target='sentiment'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    d = {'positive':1,'negative':0}
    
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

    # Apply preprocessing to the specified column
    df = df.dropna()
    df[col] = df[col].apply(preprocess_text)
    df[target] = df[target].replace({'positive':1,'negative':0})

    # Remove small sentences (less than 3 words)
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        config = load_config('config.yaml')
        # Fetch the data from data/raw
        input_path = config['data_preprocessing']['input_path']
        train_data = pd.read_csv(os.path.join(input_path,'train.csv'))
        test_data = pd.read_csv(os.path.join(input_path,'test.csv'))
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data,col= 'review',target='sentiment')
        test_processed_data = preprocess_dataframe(test_data,col = 'review',target='sentiment')

        # Store the data inside data/processed
        data_path = config['data_preprocessing']['output_path']
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()