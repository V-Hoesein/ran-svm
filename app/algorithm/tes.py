import os
import pandas as pd
import numpy as np
import joblib
import re
import joblib
import math
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Define file paths
dataset_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'dataset.csv'))
result_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads'))
model_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'model.pkl'))
tfidf_model_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'tfidf_model.pkl'))


class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Sastrawi Stemmer dan StopWordRemover
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

    def clean_text(self, text):
        # Menghapus karakter khusus dan angka, serta mengubah menjadi huruf kecil
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter non-alphabetic
        text = text.lower()  # Mengubah teks menjadi huruf kecil
        return text

    def preprocess(self, text):
        # Melakukan text cleansing, menghapus stopwords dan stemming
        text = self.clean_text(text)  # Membersihkan teks
        text = self.stopword_remover.remove(text)  # Menghapus stopwords
        text = self.stemmer.stem(text)  # Melakukan stemming
        return text

    def preprocess_text_list(self, text_list):
        # Preprocessing untuk list of text
        return [self.preprocess(text) for text in text_list]

    def preprocess_from_csv(self, csv_file):
        # Membaca file CSV dan memproses kolom 'comment'
        data = pd.read_csv(csv_file)
        data['processed_comment'] = data['comment'].apply(self.preprocess)
        return data[['comment', 'processed_comment']]

class TFIDFTest:
    def __init__(self, training_tfidf):
        self.terms = training_tfidf.terms
        self.idf = training_tfidf.idf

    def compute_tf(self, document):
        # Hitung TF untuk setiap term di dokumen
        tf = {}
        words = document.split()
        doc_len = len(words)
        print('doc_len : ', doc_len)
        for term in self.terms:
            tf[term] = words.count(term) / doc_len if doc_len > 0 else 0
        return tf

    def compute_raw_tf(self, document):
        # Hitung raw TF untuk setiap term di dokumen
        raw_tf = {}
        words = document.split()
        for term in self.terms:
            raw_tf[term] = words.count(term)  # Frekuensi mentah
        return raw_tf

    def compute_tfidf(self, test_document):
        # Hitung TF-IDF untuk dokumen tes
        tf = self.compute_tf(test_document)
        tfidf = {}
        for term in self.terms:
            tfidf[term] = tf[term] * self.idf.get(term, 0)  # Ambil IDF dengan default 0 jika term tidak ada
        return tfidf

    def get_tfidf_matrix(self, test_documents):
        # Mengembalikan matriks TF-IDF untuk semua dokumen tes
        tfidf_matrix = []
        for doc in test_documents:
            tfidf_values = self.compute_tfidf(doc)
            tfidf_matrix.append([tfidf_values.get(term, 0) for term in self.terms])  # Ambil nilai TF-IDF untuk setiap term
        return np.array(tfidf_matrix)  # Kembalikan sebagai numpy array

    def export_to_csv(self, test_documents, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            header = ['Term'] + [f'RawTF{i + 1}' for i in range(len(test_documents))] + [f'TFIDF{i + 1}' for i in range(len(test_documents))]
            writer.writerow(header)

            # Data rows
            for term in self.terms:
                row = [term]
                raw_tf_values = [self.compute_raw_tf(doc)[term] for doc in test_documents]
                tfidf_values = [self.compute_tfidf(doc)[term] for doc in test_documents]
                row.extend(raw_tf_values)  # Add raw TF values
                row.extend(tfidf_values)    # Add TF-IDF values
                writer.writerow(row)

class ManualTFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.terms = self.extract_terms()
        self.tf_matrix = self.compute_tf()
        self.df = self.compute_df()
        self.idf = self.compute_idf()
        self.tfidf_matrix = self.compute_tfidf()

    def extract_terms(self):
        # Ekstraksi semua terms unik dari seluruh dokumen
        terms = set()
        for doc in self.documents:
            terms.update(doc.split())
        return sorted(terms)

    def compute_tf(self):
    # Hitung TF untuk setiap term di setiap dokumen
        tf_matrix = []
        for doc in self.documents:
            term_count = {}
            words = doc.split()
            doc_len = len(words)
            for term in self.terms:
                term_count[term] = words.count(term)  # Frekuensi mentah
            tf_matrix.append(term_count)
        return tf_matrix

    def compute_df(self):
        # Hitung DF (Document Frequency) untuk setiap term
        df = {term: 0 for term in self.terms}
        for term in self.terms:
            df[term] = sum(1 for doc in self.documents if term in doc.split())
        return df

    def compute_idf(self):
        # Hitung IDF berdasarkan DF
        N = len(self.documents)
        idf = {term: (math.log((1 + N) / (df + 1)) + 1) for term, df in self.df.items()}
        return idf

    def compute_tfidf(self):
        # Hitung TF-IDF dengan mengalikan TF Normalized dan IDF
        tfidf_matrix = []
        normalized_tf = self.compute_normalized_tf()  # Ambil nilai TF Normalized
        for norm_tf_doc in normalized_tf:
            tfidf_doc = {}
            for term in norm_tf_doc.keys():
                tfidf_doc[term] = norm_tf_doc[term] * self.idf[term]  # Mengalikan TF Normalized dengan IDF
            tfidf_matrix.append(tfidf_doc)
        return tfidf_matrix

    def compute_normalized_tf(self):
        # Hitung TF Normalisasi
        normalized_tf_matrix = []
        for tf_doc in self.tf_matrix:
            tf_norm_doc = {}
            total_terms = sum(tf_doc.values()) if tf_doc else 1
            for term in tf_doc.keys():
                tf_norm_doc[term] = tf_doc[term] / total_terms if total_terms > 0 else 0
            normalized_tf_matrix.append(tf_norm_doc)
        return normalized_tf_matrix


    def export_to_csv(self, filename):
        normalized_tf = self.compute_normalized_tf()
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            header = ['Term']
            header += [f'TFD{i + 1}' for i in range(len(self.documents))]
            header += [f'TFNorm{i + 1}' for i in range(len(self.documents))]
            header.append('DF')
            header.append('IDF')  # Add IDF to the header
            header += [f'TFIDF{i + 1}' for i in range(len(self.documents))]
            writer.writerow(header)

            # Data rows
            for term in self.terms:
                row = [term]
                row += [self.tf_matrix[doc_idx][term] for doc_idx in range(len(self.documents))]
                row += [normalized_tf[doc_idx][term] for doc_idx in range(len(self.documents))]
                row.append(self.df[term])
                row.append(self.idf[term])  # Add IDF value for the term
                row += [self.tfidf_matrix[doc_idx][term] for doc_idx in range(len(self.documents))]
                writer.writerow(row)


    def get_tfidf_matrix(self):
        # Mengembalikan matriks TF-IDF sebagai numpy array
        return np.array([[self.tfidf_matrix[i][term] for term in self.terms] for i in range(len(self.tfidf_matrix))])
    
    def save_model(self, filepath):
        # Simpan model ke file .pkl
        model_data = {
            'terms': self.terms,
            'tf_matrix': self.tf_matrix,
            'df': self.df,
            'idf': self.idf,
            'tfidf_matrix': self.tfidf_matrix
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        # Muat model dari file .pkl
        model_data = joblib.load(filepath)
        self.terms = model_data['terms']
        self.tf_matrix = model_data['tf_matrix']
        self.df = model_data['df']
        self.idf = model_data['idf']
        self.tfidf_matrix = model_data['tfidf_matrix']

class SVM:
    def __init__(self, learning_rate=1.0, regularization_strength=0.1, n_iters=3000):
        self.lr = learning_rate
        self.reg_strength = regularization_strength
        self.n_iters = n_iters
        self.w = None  # Bobot
        self.b = None  # Bias
        self.terms = None  # Daftar term

    def fit(self, X, y, terms):
        n_samples, n_features = X.shape
        # Inisialisasi bobot dan bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Simpan daftar term
        self.terms = terms

        # Label -1 dan 1 untuk kelas negatif dan positif
        y_ = np.where(y < 0, -1, 1)

        # Pelatihan
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Hitung margin
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Jika kondisi terpenuhi, hanya update bobot
                    self.w -= self.lr * (2 * self.reg_strength * self.w)
                else:
                    # Jika tidak terpenuhi, update bobot dan bias
                    self.w -= self.lr * (2 * self.reg_strength * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Menghitung prediksi
        linear_output = np.dot(X, self.w) + self.b
        predictions = np.sign(linear_output)
        predictions[predictions == 0] = 1  # Hindari prediksi 0
        return predictions

    def export_weights_to_csv(self, filename):
        if self.terms is None or self.w is None:
            raise ValueError("Model has not been trained. Train the model before exporting weights.")
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(['Term', 'Weight'])
            # Data rows
            for term, weight in zip(self.terms, self.w):
                writer.writerow([term, weight])
        print(f"Weights exported to {filename}")

    def save_model(self, file_path):
        # Menyimpan model ke file .pkl
        joblib.dump((self.w, self.b, self.terms), file_path)

    def load_model(self, file_path):
        # Memuat model dari file .pkl
        self.w, self.b, self.terms = joblib.load(file_path)


def predict(test_documents: list):
    # Preprocessing instance
    preprocessor = TextPreprocessor()
   
    # Model instance
    svm_model = SVM()

    # Check if model and TF-IDF model already exist
    if os.path.exists(model_path) and os.path.exists(tfidf_model_path):
        # Load the trained SVM model and TF-IDF model
        svm_model.w, svm_model.b = joblib.load(model_path)
        
        # Load the TF-IDF model
        tfidf_model = ManualTFIDF([])
        tfidf_model.load_model(tfidf_model_path)
        print("Model and TF-IDF model loaded from", model_path, "and", tfidf_model_path)
    else:
        # If model does not exist, train the model
        processed_data = preprocessor.preprocess_from_csv(dataset_path)
        preprocessed_text = list(processed_data['processed_comment'])

        # Build TFIDF model
        data = pd.read_csv(dataset_path)
        tfidf_model = ManualTFIDF(preprocessed_text)
        tfidf_model.export_to_csv(f'{result_path}/tfidf_result.csv')
        
        # Get labels for training
        y_tfidf = list(data['label'])
        
        # Train SVM model
        X_train = tfidf_model.get_tfidf_matrix()  # Get TF-IDF matrix from training data
        y_train = np.array(y_tfidf)  # Convert labels to numpy array
    
        # Fit SVM model to training data
        svm_model.fit(X_train, y_train, tfidf_model.terms)  # Use terms from the TF-IDF model
        svm_model.export_weights_to_csv(f'{result_path}/decision_function.csv')
        
        # Save trained model and TF-IDF model
        joblib.dump((svm_model.w, svm_model.b), model_path)
        tfidf_model.save_model(tfidf_model_path)  # Save the TF-IDF model
        print("Model and TF-IDF model trained and saved to", model_path, "and", tfidf_model_path)

    # Lakukan preprocessing text test
    preprocessed_test = preprocessor.preprocess_text_list(test_documents)
    
    # Process test documents with TF-IDF
    tfidf_test = TFIDFTest(tfidf_model)  # Pass the TFIDF object directly
    tfidf_test.export_to_csv(preprocessed_test, f'{result_path}/tfidf_test_result.csv')
    
    # Calculate TF-IDF matrix for test documents
    X_test = tfidf_test.get_tfidf_matrix(test_documents)
    
    # Perform predictions on test documents
    predictions = svm_model.predict(X_test)

    print(predictions)
    predictions = ["negatif" if p < 0 else "positif" for p in predictions]
    print("Predictions:", predictions)
    return predictions


predict(['aku percaya percaya banget percaya'])