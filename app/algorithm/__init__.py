import os
import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads','dataset.csv'))
result_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads'))

class TextPreprocessor:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        self.combined_stopwords = set(StopWordRemoverFactory().get_stop_words()).union(set(stopwords.words('english')))

    def clean_text(self, text):
        """Remove punctuation, digits, and unwanted characters."""
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def tokenize(self, text):
        """Lowercase text and tokenize."""
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        """Remove stopwords."""
        return [word for word in tokens if word not in self.combined_stopwords]

    def stem(self, tokens):
        """Apply stemming."""
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess(self, text):
        """Pipeline for text preprocessing: clean, tokenize, remove stopwords, and stem."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(self.stem(tokens))


class TextClassifier:
    def __init__(self, vectorizer=None, model=None):
        self.vectorizer = vectorizer or TfidfVectorizer()
        self.model = model or svm.SVC(random_state=0, kernel='rbf')
        self.result_path = result_path

    def calculate_raw_tf(self, X_train_vec):
        """Calculate Term Frequency (TF) without normalization."""
        return X_train_vec.toarray()

    def calculate_tf_norm(self, X_train_vec):
        """Calculate Term Frequency Normalized (TF Norm)."""
        tf_array = X_train_vec.toarray()
        tf_norm = tf_array / np.linalg.norm(tf_array, axis=1, keepdims=True)  # Normalized TF
        tf_norm = np.nan_to_num(tf_norm)  # Handle potential NaN values
        return tf_norm

    def calculate_df_idf(self, X_train_vec):
        """Calculate Document Frequency (DF) and Inverse Document Frequency (IDF)."""
        df = np.sum(X_train_vec.toarray() > 0, axis=0)  # Count of documents containing each term (DF)
        idf = np.log((1 + X_train_vec.shape[0]) / (1 + df)) + 1  # Calculate IDF
        return df, idf

    def create_export_dataframe(self, X_vec, tf_norm, df, idf):
        """Create DataFrame for exporting TF, TFNorm, DF, IDF, and TF-IDF."""
        # Get TF, DF, and IDF values
        tf_array = X_vec.toarray()
        tfidf_array = tf_array * idf  # Calculate TF-IDF by multiplying TF with IDF

        # Create DataFrame with Terms as rows
        tfidf_df = pd.DataFrame(tf_array, columns=self.vectorizer.get_feature_names_out())
        tfidf_transposed = tfidf_df.T  # Transpose the TF matrix
        
        # Add extra columns: TF Norm, DF, IDF, TF-IDF
        tfidf_transposed['TFNormAll'] = tf_norm.sum(axis=0)  # Sum TF Norm across all documents for each term
        tfidf_transposed['DF'] = df  # Document Frequency
        tfidf_transposed['IDF'] = idf  # Inverse Document Frequency
        tfidf_transposed['TFIDF'] = tfidf_array.sum(axis=0)  # Sum TF-IDF values for each term

        # Set the document names as columns
        tfidf_transposed.columns = [f'D{i+1}' for i in range(X_vec.shape[0])] + ['TFNormAll', 'DF', 'IDF', 'TFIDF']
        tfidf_transposed.index.name = 'Terms'  # Set the index name as 'Terms'

        # Round all numeric values to 3 decimal places
        tfidf_transposed = tfidf_transposed.round(3)

        return tfidf_transposed

    def train(self, X_train, y_train):
        """Train the model with the training data."""
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Calculate TF, TF Norm, DF, and IDF
        tf = self.calculate_raw_tf(X_train_vec)
        tf_norm = self.calculate_tf_norm(X_train_vec)
        df, idf = self.calculate_df_idf(X_train_vec)

        # Create DataFrame for export
        tfidf_train_transposed = self.create_export_dataframe(X_train_vec, tf_norm, df, idf)

        # Export the training results to CSV
        tfidf_train_transposed.to_csv(f'{self.result_path}/tfidf_train_with_metrics.csv', index=True)

        # Train the model
        self.model.fit(X_train_vec, y_train)

    def predict(self, X_test):
        """Predict the labels of the test set."""
        X_test_vec = self.vectorizer.transform(X_test)

        # Calculate TF Norm, DF, and IDF for test set
        tf = self.calculate_raw_tf(X_test_vec)
        tf_norm = self.calculate_tf_norm(X_test_vec)
        df, idf = self.calculate_df_idf(X_test_vec)

        # Create DataFrame for export
        tfidf_test_transposed = self.create_export_dataframe(X_test_vec, tf_norm, df, idf)

        # Export the test results to CSV
        tfidf_test_transposed.to_csv(f'{self.result_path}/tfidf_test_with_metrics.csv', index=True)

        return self.model.predict(X_test_vec)

    def evaluate(self, y_test, y_pred):
        """Evaluate the model using common metrics."""
        results = {
            'F1-Score': f1_score(y_test, y_pred, pos_label='positif'),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label='positif'),
            'Recall': recall_score(y_test, y_pred, pos_label='positif'),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }

        # Export evaluation metrics to CSV
        eval_df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
        eval_df.to_csv(f'{self.result_path}/evaluation_metrics.csv')

        return results


def predict(text_test: str):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Initialize preprocessor and preprocess the comments
    preprocessor = TextPreprocessor()
    df['preprocessed_comment'] = df['comment'].apply(preprocessor.preprocess)

    # Split the data into training and testing sets
    X_train = df['preprocessed_comment']
    y_train = df['label']

    # Initialize classifier
    classifier = TextClassifier()

    # Train the model
    classifier.train(X_train, y_train)

    # Preprocess the input text
    text_test_preprocessed = preprocessor.preprocess(text_test)

    # Predict the label for the input text
    prediction = classifier.predict([text_test_preprocessed])

    # Return the prediction result
    return prediction[0]
