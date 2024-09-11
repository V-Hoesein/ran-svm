import os
import string
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.svm import SVC

class TextClassifier:
    def __init__(self, dataset_path, result_path):
        self.dataset_path = dataset_path
        self.result_path = result_path
        
        # Initialize Sastrawi tools
        self.stemmer = StemmerFactory().create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.combined_stopwords = set(stopword_factory.get_stop_words()).union(set(stopwords.words('english')))
        
        # Load dataset
        self.df_comments = pd.read_csv(self.dataset_path)
        
        # Preprocess text
        self.df_comments['preprocess'] = self.df_comments['comment'].apply(self.preprocess_text)
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
    
    def clean_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text).lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.combined_stopwords]
        stemmed = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed)
    
    def compute_raw_tf(self, doc):
        words = doc.split()
        count = Counter(words)
        return count
    
    def compute_tf(self, doc):
        words = doc.split()
        count = Counter(words)
        total_terms = len(words)
        tf = {term: count[term] / total_terms for term in count}
        return tf
    
    def train_model(self):
        # Fit the vectorizer on training data
        X_train = self.vectorizer.fit_transform(self.df_comments['preprocess'])
        y_train = self.df_comments['label']
        
        # Get TF-IDF terms and IDF values
        terms = self.vectorizer.get_feature_names_out()
        idf_values = self.vectorizer.idf_
        
        # Create DataFrame for TF-IDF
        tfidf_df = pd.DataFrame(X_train.toarray().T, index=terms, columns=[f'D{i+1}' for i in range(len(self.df_comments['preprocess']))])
        idf_df = pd.DataFrame(idf_values, index=terms, columns=["IDF"])
        
        # Compute raw TF for each document
        raw_tf_dicts = [self.compute_raw_tf(doc) for doc in self.df_comments['preprocess']]
        raw_tf_df = pd.DataFrame(raw_tf_dicts, index=[f'D{i+1}' for i in range(len(self.df_comments['preprocess']))]).T
        
        # Compute normalized TF for each document
        tf_dicts = [self.compute_tf(doc) for doc in self.df_comments['preprocess']]
        tf_df = pd.DataFrame(tf_dicts, index=[f'D{i+1}' for i in range(len(self.df_comments['preprocess']))]).T
        
        # Fill NaN values with 0
        raw_tf_df = raw_tf_df.fillna(0)
        tf_df = tf_df.fillna(0)
        idf_df = idf_df.fillna(0)
        tfidf_df = tfidf_df.fillna(0)
        
        # Sum all normalized TF values across all documents
        tf_norm_all = tf_df.sum(axis=1)
        
        # Compute Document Frequency (DF)
        df_values = (raw_tf_df > 0).sum(axis=1)
        
        # Create final DataFrame
        final_df = pd.DataFrame(index=terms)
        final_df['Terms'] = terms
        final_df = final_df.join(raw_tf_df.add_prefix('TF'))
        final_df = final_df.join(tf_df.add_prefix('TFN'))
        final_df['TFNAll'] = tf_norm_all
        final_df['DF'] = df_values
        final_df['IDF'] = idf_df['IDF']
        final_df = final_df.join(tfidf_df.add_prefix('TFIDF_'))
        final_df = final_df.round(3)
        
        # Export final DataFrame to CSV
        final_df.to_csv(f'{self.result_path}/train_metrics.csv', index=False)
        
        # Train the SVM model
        self.model = SVC(random_state=0, kernel='linear')
        self.model.fit(X_train, y_train)
    
    def predict(self, text_test: str):
        X_test = self.preprocess_text(text_test)
        X_test_vectorized = self.vectorizer.transform([X_test])
        prediction = self.model.predict(X_test_vectorized)
        return prediction[0]

def predict(new_text:str):
    # Usage
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads'))

    classifier = TextClassifier(dataset_path, result_path)
    classifier.train_model()

    # Make predictions
    prediction = classifier.predict(new_text)
    return prediction