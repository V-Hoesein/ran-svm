import string
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords

class TextCleaner:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.combined_stopwords = set(stopword_factory.get_stop_words()).union(set(stopwords.words('english')))

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


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class TFIDFVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def compute_metrics(self, df_comments):
        # Fit the vectorizer on training data to get IDF values
        self.fit(df_comments['preprocess'])
        idf_values = self.vectorizer.idf_
        terms = self.vectorizer.get_feature_names_out()

        raw_tf_dicts = [self.compute_raw_tf(doc) for doc in df_comments['preprocess']]
        tf_dicts = [self.compute_tf(doc) for doc in df_comments['preprocess']]

        raw_tf_df = pd.DataFrame(raw_tf_dicts).fillna(0).T
        tf_df = pd.DataFrame(tf_dicts).fillna(0).T
        idf_df = pd.DataFrame(idf_values, index=terms, columns=["IDF"])

        # Compute Document Frequency (DF)
        df_values = (raw_tf_df > 0).sum(axis=1)

        # Create final DataFrame
        final_df = pd.DataFrame(index=terms)
        final_df['Term'] = terms  # Add 'Term' column
        final_df['DF'] = df_values
        final_df['IDF'] = idf_df['IDF']

        # Add raw TF and normalized TF to final DataFrame
        final_df = final_df.join(raw_tf_df.add_prefix('TF'))
        final_df = final_df.join(tf_df.add_prefix('TFN'))

        # Compute manual TF-IDF (TFN * IDF)
        for doc in tf_df.columns:
            final_df[f'TFIDF_{doc}'] = final_df[f'TFN{doc}'] * final_df['IDF']

        return final_df.round(3)

import joblib
import os
from sklearn.svm import SVC

class TextClassifier:
    def __init__(self, dataset_path, result_path):
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.cleaner = TextCleaner()
        self.vectorizer = TFIDFVectorizer()

        # Load dataset
        self.df_comments = pd.read_csv(self.dataset_path)
        self.df_comments['preprocess'] = self.df_comments['comment'].apply(self.cleaner.preprocess_text)

    def train_model(self):
        # Compute TF-IDF metrics
        final_df = self.vectorizer.compute_metrics(self.df_comments)

        # Save metrics to CSV
        final_df.to_csv(f'{self.result_path}/train_metrics.csv', index=False)

        # Train SVM model
        X_train = self.vectorizer.transform(self.df_comments['preprocess'])
        y_train = self.df_comments['label']
        self.model = SVC(random_state=0, kernel='linear')
        self.model.fit(X_train, y_train)

    def predict(self, text_test: str):
        X_test = self.cleaner.preprocess_text(text_test)
        X_test_vectorized = self.vectorizer.transform([X_test])
        prediction = self.model.predict(X_test_vectorized)
        return prediction[0]

class ModelManager:
    def __init__(self, dataset_path, result_path):
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.model_file = os.path.join(result_path, 'trained_model.pkl')

    def load_or_train_model(self):
        if os.path.exists(self.model_file):
            return joblib.load(self.model_file)
        else:
            classifier = TextClassifier(self.dataset_path, self.result_path)
            classifier.train_model()
            joblib.dump(classifier, self.model_file)
            return classifier


def predict(new_text: str):
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads'))

    model_manager = ModelManager(dataset_path, result_path)
    classifier = model_manager.load_or_train_model()

    # Preprocess and transform the new_text to get TF-IDF values
    preprocessed_text = classifier.cleaner.preprocess_text(new_text)
    tfidf_vector = classifier.vectorizer.transform([preprocessed_text])

    # Get feature names
    feature_names = classifier.vectorizer.vectorizer.get_feature_names_out()
    
    # Create a DataFrame for the TF-IDF values
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=feature_names)

    # Export TF-IDF values to CSV
    tfidf_df.to_csv(f'{result_path}/tfidf_values.csv', index=False)

    # Make predictions
    prediction = classifier.predict(new_text)

    return prediction
