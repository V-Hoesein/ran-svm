import os
import pandas as pd
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
        self.model = model or svm.SVC(kernel='linear')

    def train(self, X_train, y_train):
        """Train the model with the training data."""
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def predict(self, X_test):
        """Predict the labels of the test set."""
        X_test_vec = self.vectorizer.transform(X_test)
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
        return results


def main():
    # Load dataset
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'dataset-10.csv'))
    df = pd.read_csv(dataset_path)

    # Initialize preprocessor and preprocess the comments
    preprocessor = TextPreprocessor()
    df['preprocessed_comment'] = df['comment'].apply(preprocessor.preprocess)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_comment'], df['label'], test_size=0.2, stratify=df['label'], random_state=20)

    # Initialize classifier and vectorizer
    classifier = TextClassifier()

    # Train the model
    classifier.train(X_train, y_train)

    # Predict the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    results = classifier.evaluate(y_test, y_pred)

    # Output the results
    for metric, score in results.items():
        print(f'{metric}: {score}')

    new_comment = preprocessor.preprocess('jelek')
    print(classifier.predict([new_comment]))
    


if __name__ == '__main__':
    main()
