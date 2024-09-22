import csv
import os
import math
from collections import Counter
import string
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

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

    def preprocess_text(self, text) -> str:
        text = self.clean_text(text).lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.combined_stopwords]
        stemmed = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed)


class TFIDFCalculator:
    def __init__(self, input_file, output_file, text_cleaner):
        self.input_file = input_file
        self.output_file = output_file
        self.text_cleaner = text_cleaner
        self.documents = []
        self.terms = set()

    def load_data(self):
        with open(self.input_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cleaned_text = self.text_cleaner.preprocess_text(row['comment'])
                self.documents.append(cleaned_text)
        self.terms = set(' '.join(self.documents).split())

    def calculate_tf(self):
        term_frequencies = []
        for doc in self.documents:
            counter = Counter(doc.split())
            tf_raw = {term: counter.get(term, 0) for term in self.terms}
            total_terms = sum(counter.values())
            tf_norm = {term: round(count / total_terms, 4) for term, count in tf_raw.items()}
            tf_raw = {term: round(count, 4) for term, count in tf_raw.items()}
            term_frequencies.append((tf_raw, tf_norm))
        return term_frequencies

    def calculate_df(self):
        df = {term: 0 for term in self.terms}
        for doc in self.documents:
            doc_terms = set(doc.split())
            for term in doc_terms:
                df[term] += 1
        return df

    def calculate_idf(self, df):
        num_docs = len(self.documents)
        idf = {term: round(math.log((1 + num_docs) / (1+df[term]))+1, 4) for term in self.terms}
        return idf

    def calculate_tfidf(self, term_frequencies, idf):
        tfidf = []
        tfidf_norm = []
        
        for tf_raw, tf_norm in term_frequencies:
            # Calculate unnormalized TF-IDF using tf_norm
            tfidf_doc = {term: round(tf_norm[term] * idf[term], 4) for term in self.terms}
            tfidf.append(tfidf_doc)

            # Calculate L2 norm for normalization
            norm = math.sqrt(sum(value**2 for value in tfidf_doc.values()))
            
            # Normalize using L2 norm
            if norm != 0:
                tfidf_norm_doc = {term: round(value / norm, 4) for term, value in tfidf_doc.items()}
            else:
                tfidf_norm_doc = {term: 0 for term in self.terms}
            
            tfidf_norm.append(tfidf_norm_doc)
        
        return tfidf, tfidf_norm


    def export_results(self, term_frequencies, df, idf, tfidf, tfidf_norm):
        with open(self.output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Term'] + [f'TF{i + 1}' for i in range(len(self.documents))] + \
                    [f'TFN{i + 1}' for i in range(len(self.documents))] + ['DF', 'IDF'] + \
                    [f'TFIDF{i + 1}' for i in range(len(self.documents))] + \
                    [f'TFIDFN{i + 1}' for i in range(len(self.documents))]

            writer.writerow(header)

            # Sort the terms in ascending order
            sorted_terms = sorted(self.terms)

            for term in sorted_terms:
                row = [term]
                for doc_tf in term_frequencies:
                    row.append(doc_tf[0][term])
                for doc_tf in term_frequencies:
                    row.append(doc_tf[1][term])
                row.append(df[term])
                row.append(idf[term])
                for doc_tfidf in tfidf:
                    row.append(doc_tfidf[term])
                for doc_tfidf_norm in tfidf_norm:
                    row.append(doc_tfidf_norm[term])
                writer.writerow(row)

    def process(self):
        self.load_data()
        term_frequencies = self.calculate_tf()
        df = self.calculate_df()
        idf = self.calculate_idf(df)
        tfidf, tfidf_norm = self.calculate_tfidf(term_frequencies, idf)
        self.export_results(term_frequencies, df, idf, tfidf, tfidf_norm)
        return tfidf_norm


#* Fungsi untuk generate csv metriks tfidf dan svm model
def train_model():
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'tfidf.csv'))
    model_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'model.pkl'))
    
    # Mengambil label pada dataset csv
    data_csv = pd.read_csv(dataset_path)
    documents_sentiment = list(data_csv['label']) 

    # Menghitung TFIDF
    text_cleaner = TextCleaner()
    
    tfidf_calculator = TFIDFCalculator(dataset_path, result_path, text_cleaner)
    metricts_tfidf = tfidf_calculator.process()

    # Transformasi array of dict tfidf kedalam dataframe
    X = pd.DataFrame(metricts_tfidf)

    # Konversi label sentimen menjadi angka 0 untuk negatif dan 1 untuk positif
    encoder = LabelEncoder()
    y = encoder.fit_transform(documents_sentiment)

    # Membuat Model SVM / Train
    model = SVC(kernel='linear')

    model.fit(X,y)

    joblib.dump(model, model_path)
    

def predict(test_text:str):
    # Load dataset
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'dataset.csv'))
    data = pd.read_csv(dataset_path)
    raw_data = list(data['comment'])

    preprocessor = TextCleaner()
    data_train = [preprocessor.preprocess_text(raw) for raw in raw_data]

    # Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data_train)

    # Convert to DataFrame
    terms = vectorizer.get_feature_names_out()  # Get terms from the vectorizer
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)

    # Save to CSV in ascending order by term
    tfidf_df_sorted = tfidf_df.sort_index(axis=1)
    tfidf_df_sorted.to_csv('tfidf_from_lib.csv', index=False)

    # Prepare data testing
    data_test = preprocessor.preprocess_text(test_text)

    # FIT data testing
    tfidf_matrix_test = vectorizer.transform([data_test])
    
    tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=terms)
    
    # Save to CSV in ascending order by term and transpose
    tfidf_df_sorted_test = tfidf_df_test.sort_index(axis=1).T
    tfidf_df_sorted_test.columns = ['TFIDFN']  # Beri nama kolom setelah transposisi
    tfidf_df_sorted_test.to_csv('tfidf_fit_from_lib.csv', index_label='Terms')



predict('cantiknya tasya farasya kebangetan.??semoga sehat selalu')
