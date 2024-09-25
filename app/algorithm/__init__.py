import csv
import os
import math
from collections import Counter
import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import joblib
import time

from sklearn.feature_extraction.text import TfidfVectorizer

class TextCleaner:
    def __init__(self):
        print('===== Preprocessing =====')
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
        print('===== Menghitung TFIDF Manual =====')
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


#* Fungsi untuk generate csv metriks tfidf
def generate_csv_manual():
    print('===== Membuat .csv =====')
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'tfidf.csv'))
    
    # Mengambil label pada dataset csv
    data_csv = pd.read_csv(dataset_path)
    documents_sentiment = list(data_csv['label'])

    text_cleaner = TextCleaner()
    
    tfidf_calculator = TFIDFCalculator(dataset_path, result_path, text_cleaner)
    metricts_tfidf = tfidf_calculator.process()
    
    print('TFIDF has calculated!')
    return metricts_tfidf


#* Fungsi untuk train model SVM
def train_model():
    print('===== Melatih Model =====')
    
    # Path file
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'model.pkl'))
    X_train_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'X_train.pkl'))
    encoder_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'encoder.pkl'))
    
    # Load dataset
    data_csv = pd.read_csv(dataset_path)
    documents_sentiment = list(data_csv['label'])

    # Preprocessing teks
    text_cleaner = TextCleaner()
    cleaned_data = [text_cleaner.preprocess_text(text) for text in data_csv['comment']]
    
    # Menghitung TFIDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(cleaned_data)
    
    # Simpan vectorizer
    joblib.dump(vectorizer, X_train_path)
    
    # Label encoding (ubah label menjadi angka)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(documents_sentiment)

    # Simpan encoder
    joblib.dump(encoder, encoder_path)

    # Membuat dan melatih model SVM dengan kernel linear
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Simpan model yang sudah dilatih
    joblib.dump(model, model_path)

    print("Model, TFIDF vectorizer, dan encoder berhasil disimpan.")


#* Fungsi untuk evaluasi model
def evaluate_model():
    start_time = time.time()
    
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'dataset.csv'))
    X_train_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'X_train.pkl'))
    encoder_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'encoder.pkl'))
    model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'model.pkl'))

    # Load dataset
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)

    # Load vectorizer and transform the data
    print("Loading vectorizer and transforming data...")
    vectorizer = joblib.load(X_train_path)
    X = vectorizer.transform(data['comment'])  # Transforming to sparse matrix

    # Load encoder and encode labels
    print("Loading encoder and transforming labels...")
    encoder = joblib.load(encoder_path)
    y = encoder.transform(data['label'])

    # Split data into training and testing sets (80:20)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load pre-trained model or train it if not available
    print("Loading or training the model...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    
    # Melakukan prediksi pada test set
    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    # Menghitung metrik evaluasi
    print("Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Menampilkan hasil evaluasi
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    # Print total time taken
    print(f"--- Total evaluation time: {time.time() - start_time:.2f} seconds ---")
    
    return accuracy, f1, recall, precision


#* Fungsi untuk melihat hyperlance
def print_hyperplane(model):
    print('===== Hyperlance =====')
    # Akses nilai w dan b dari model
    w = model.coef_.toarray()  # konversi sparse matrix ke dense array
    b = model.intercept_[0]  # bias
    
    print(f"Persamaan hyperplane: w·x + b = 0")
    print(f"w (vektor bobot): {w}")
    print(f"b (bias): {b}")
    
    # Cek dimensi w dengan menggunakan .shape, bukan len()
    if w.shape[1] == 2:  # Jika hanya dua fitur (2D), kita bisa cetak atau visualisasikan hyperplane
        print(f"Persamaan hyperplane dalam 2D: {w[0][0]}*x1 + {w[0][1]}*x2 + {b} = 0")

    return w, b


#* Fungsi untuk prediksi
def predict(test_text: str):
    print('===== Prediksi =====')
    X_train_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..',  'static', 'uploads', 'X_train.pkl'))
    model_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  '..', 'static', 'uploads', 'model.pkl'))
    output_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads'))

    # Preprocessing test text
    preprocessor = TextCleaner()
    test_text = preprocessor.preprocess_text(test_text)

    # Vectorization
    tfidf_model = joblib.load(X_train_path)
    tfidf_matrix_test = tfidf_model.transform([test_text])
    
    #! SVM
    model = joblib.load(model_path)
    
    # Predict
    prediction = model.predict(tfidf_matrix_test)

    # Akses nilai w, b, dan x
    w = model.coef_.toarray()  # vektor bobot
    b = model.intercept_  # bias
    x = tfidf_matrix_test.toarray()  # vektor input data (dikonversi ke dense array)
    
    # Simpan w, b, dan x ke file CSV terpisah
    pd.DataFrame(w).to_csv(os.path.join(output_dir, 'w.csv'), header=False, index=False)
    pd.DataFrame(b).to_csv(os.path.join(output_dir, 'b.csv'), header=False, index=False)
    pd.DataFrame(x).to_csv(os.path.join(output_dir, 'x.csv'), header=False, index=False)
    
    
    # ** Export ke .csv hasil Fit test text dengan dokumen latih
    terms = tfidf_model.get_feature_names_out()
    tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=terms)

    # Hitung TF mentah
    raw_counts = pd.Series(test_text.split()).value_counts()
    tf_values = [raw_counts.get(term, 0) for term in terms]
    
    # Tambahkan kolom TF mentah
    tfidf_df_sorted_test = tfidf_df_test.sort_index(axis=1).T
    tfidf_df_sorted_test['TF'] = tf_values
    
    # Menyesuaikan urutan kolom
    tfidf_df_sorted_test = tfidf_df_sorted_test[['TF', 0]]

    # Simpan ke CSV
    tfidf_df_sorted_test.columns = ['TF', 'TFIDFN']
    tfidf_df_sorted_test.to_csv(f'{output_dir}/tfidf_fit.csv', index_label='Terms')

    return prediction



# generate_csv_manual()
# train_model()
# print(predict('cantiknya tasya farasya kebangetan.??semoga sehat selalu'))
# print(predict('Sumpah ya Avoskin cica mugwort gue pKe malah jerawatan.'))
# print(predict('Tidak sesuai sama yng d pesen, yang d pesen sadeor yng datang hymeys.... Sangat tidak sesuai sama yang d pesen....'))
evaluate_model()