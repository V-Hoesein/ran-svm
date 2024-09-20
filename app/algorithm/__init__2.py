import os
import string
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import joblib

dataset_path = os.path.realpath(os.path.join(os.path.dirname(__name__), '..', 'app','static','uploads', 'dataset.csv'))
result_path = os.path.realpath(os.path.join(os.path.dirname(__name__), '..', 'notebook', 'result'))

class TFIDFProcessor:
    def __init__(self):
        # Initialize Sastrawi tools
        self.stemmer = StemmerFactory().create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.combined_stopwords = set(stopword_factory.get_stop_words()).union(set(stopwords.words('english')))
        self.terms = None  # To store terms after processing corpus

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
        tf = {term: float(count[term]) / total_terms for term in count}
        return tf

    def compute_idf(self, corpus):
        N = len(corpus)
        idf_dict = {}
        all_words = set(word for doc in corpus for word in doc.split())
        
        for word in all_words:
            containing_docs = sum(1 for doc in corpus if word in doc.split())
            idf_dict[word] = float(np.log((1+N) / (1 + containing_docs))+1)  # Make sure IDF is a float
        
        return idf_dict

    def compute_tfidf(self, tf_dict, idf_dict):
        tfidf_dict = {}
        for word, tf_value in tf_dict.items():
            tfidf_dict[word] = float(tf_value) * float(idf_dict.get(word, 0.0))  # Ensure both are floats
        
        return tfidf_dict

    def process_corpus(self, corpus):
        # Compute IDF for the corpus
        idf_values = self.compute_idf(corpus)
        
        # Get the terms (features)
        self.terms = sorted(idf_values.keys())
        
        # Compute raw TF, normalized TF, and TF-IDF for each document
        raw_tf_dicts = [self.compute_raw_tf(doc) for doc in corpus]
        tf_dicts = [self.compute_tf(doc) for doc in corpus]
        
        # Ensure each document's TF-IDF vector contains all terms
        tfidf_dicts = []
        for tf_dict in tf_dicts:
            tfidf_dict = {}
            for term in self.terms:
                tfidf_dict[term] = tf_dict.get(term, 0) * idf_values.get(term, 0)
            tfidf_dicts.append(tfidf_dict)
        
        # Convert dictionaries to DataFrames for easy manipulation and export
        raw_tf_df = pd.DataFrame(raw_tf_dicts, index=[f'D{i+1}' for i in range(len(corpus))]).T
        tf_df = pd.DataFrame(tf_dicts, index=[f'D{i+1}' for i in range(len(corpus))]).T
        tfidf_df = pd.DataFrame(tfidf_dicts, index=[f'D{i+1}' for i in range(len(corpus))]).T
        
        # Fill NaN values with 0
        raw_tf_df = raw_tf_df.fillna(0)
        tf_df = tf_df.fillna(0)
        tfidf_df = tfidf_df.fillna(0)
        
        return self.terms, raw_tf_df, tf_df, tfidf_df, idf_values


class SVMClassifier:
    def __init__(self):
        self.weights = None
        self.bias = None

    def train_svm(self, X_train, y_train, lr=0.0001, epochs=1000, C=1.0):
        """
        Train a linear SVM using gradient descent.

        Parameters:
        - X_train: Training feature matrix
        - y_train: Training labels
        - lr: Learning rate
        - epochs: Number of iterations
        - C: Regularization parameter
        """
        num_samples, num_features = X_train.shape
        weights = np.zeros(num_features)
        bias = 0

        # Gradient descent for SVM
        for epoch in range(epochs):
            for i in range(num_samples):
                condition = y_train[i] * (np.dot(X_train[i], weights) - bias) >= 1
                if condition:
                    weights -= lr * (2 * C * weights)  # Regularization term
                else:
                    weights -= lr * (2 * C * weights - np.dot(X_train[i], y_train[i]))
                    bias -= lr * y_train[i]
        
        # Round weights and bias to 4 decimal places
        self.weights, self.bias = np.round(weights, 4), round(bias, 4)

    def predict(self, X):
        """
        Predict the label of a given input vector X using the learned weights and bias.
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.round(np.sign(linear_output), 4)  # Round predictions to 4 decimal places

    def compute_decision_values(self, X):
        """
        Compute decision values for each document in X using native Python.
        """
        decision_values = []
        print(f'X /n {X}')

        # Loop over each document (row in X)
        for i in range(len(X)):
            dot_product = 0
            
            # Compute dot product between the feature vector X[i] and weights
            for j in range(len(X[i])):
                dot_product += X[i][j] * self.weights[j]
                # print(f'[i:{i},j:{j}] = {X[i][j]} * {self.weights[j]} = {dot_product}')

            # Compute the decision value for the document by subtracting the bias
            decision_value = dot_product - self.bias
            
            # Round the decision value to 4 decimal places
            decision_values.append(round(decision_value, 4))

        return decision_values

    def get_weights_bias(self):
        """
        Get the rounded weights and bias.
        """
        return self.weights, self.bias


class TextClassifier:
    def __init__(self, dataset_path, result_path):
        self.dataset_path = dataset_path
        self.result_path = result_path
        
        # Initialize processors
        self.tfidf_processor = TFIDFProcessor()
        self.svm_classifier = SVMClassifier()
        
        # Load dataset
        self.df_comments = pd.read_csv(self.dataset_path)
        
        # Preprocess text
        self.df_comments['preprocess'] = self.df_comments['comment'].apply(self.tfidf_processor.preprocess_text)

    def train_model(self):
        # Preprocess all text in the dataset
        corpus = self.df_comments['preprocess'].tolist()
        
        # Process corpus for TF-IDF
        self.terms, raw_tf_df, tf_df, tfidf_df, idf_values = self.tfidf_processor.process_corpus(corpus)
        
        # Create Document Frequency (DF)
        df_values = (raw_tf_df > 0).sum(axis=1)
        
        # Create final DataFrame
        final_df = pd.DataFrame(index=self.terms)
        final_df['Terms'] = self.terms
        final_df = final_df.join(raw_tf_df.add_prefix('TF'))  # Add raw term counts for each document
        final_df = final_df.join(tf_df.add_prefix('TFN'))  # Add normalized TF for each document
        final_df = final_df.join(tfidf_df.add_prefix('TFIDF'))  # Add TF-IDF for each document
        
        # Add Document Frequency (DF) and IDF values
        final_df['DF'] = df_values
        final_df['IDF'] = [idf_values.get(term, 0) for term in self.terms]
        
        # Round all numeric columns to 4 decimal places
        final_df = final_df.round(4)
        
        # Export final DataFrame to CSV
        final_df.to_csv(f'{self.result_path}/train_metrics.csv', index=False)
        
        # Prepare training data for SVM
        X_train = np.array([list(tfidf_df.loc[:, f'D{i+1}']) for i in range(len(corpus))])
        y_train = self.df_comments['label'].values
        
        # Train SVM manually
        self.svm_classifier.train_svm(X_train, y_train)

        # Compute decision values
        decision_values = self.svm_classifier.compute_decision_values(X_train)
        print('tfidf_df :')
        print(tfidf_df)
        print('X_train :')
        print(X_train)
        print('decision_values :')
        print(decision_values)
        
        # Export decision values to CSV
        decision_values_df = pd.DataFrame(decision_values, index=[f'D{i+1}' for i in range(len(corpus))], columns=['DecisionValue'])
        decision_values_df.to_csv(f'{self.result_path}/decision_values.csv', index=True)

        # Export weights and bias to CSV
        weights, bias = self.svm_classifier.get_weights_bias()
        weights_bias_df = pd.DataFrame({'Weights': weights, 'Bias': [bias] * len(weights)}, index=[f'Feature{i+1}' for i in range(len(weights))])
        weights_bias_df.to_csv(f'{self.result_path}/weights_bias.csv', index=True)

    def predict(self, text_test):
        preprocessed_text = self.tfidf_processor.preprocess_text(text_test)
        test_tf = self.tfidf_processor.compute_tf(preprocessed_text)
        
        # Ensure the prediction uses the same set of terms from training
        idf_values = self.tfidf_processor.compute_idf([preprocessed_text])
        test_tfidf = {term: round(test_tf.get(term, 0) * idf_values.get(term, 0), 4) for term in self.tfidf_processor.terms}
        
        # Convert test_tfidf to a rounded vector
        test_vector = np.array([round(test_tfidf.get(term, 0), 4) for term in self.tfidf_processor.terms])
        
        # **Export TF-IDF values for the test document to CSV**
        test_tfidf_df = pd.DataFrame([test_tfidf], columns=self.tfidf_processor.terms).T
        test_tfidf_df.columns = ['TFIDF_Test']
        
        # Round DataFrame to 4 decimal places
        test_tfidf_df = test_tfidf_df.round(4)
        test_tfidf_df.to_csv(f'{self.result_path}/test_tfidf.csv', index=True)
        
        # Predict the label for the test document
        prediction = self.svm_classifier.predict(test_vector)

        return prediction



def predict(new_text: str):
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads'))

    # Model file path
    model_file = os.path.join(result_path, 'trained_model.pkl')

    # Check if the trained model already exists
    if os.path.exists(model_file):
        # Load the existing model
        classifier = joblib.load(model_file)
        prediction = classifier.predict(new_text)
    else:
        # Train and save the model if not exist
        classifier = TextClassifier(dataset_path, result_path)
        classifier.train_model()
        joblib.dump(classifier, model_file)

        prediction = classifier.predict(new_text)
        print(prediction)

    if prediction > 0.0:
        prediction = 'positif'
    else:
        prediction = 'negatif'
        
    return prediction
