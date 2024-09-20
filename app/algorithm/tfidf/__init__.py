import joblib
import math
import csv
import numpy as np

class TFIDFTest:
    def __init__(self, training_tfidf):
        self.terms = training_tfidf.terms
        self.idf = training_tfidf.idf

    def compute_tf(self, document):
        # Hitung TF untuk setiap term di dokumen
        tf = {}
        words = document.split()
        doc_len = len(words)
        for term in self.terms:
            tf[term] = words.count(term) / doc_len if doc_len > 0 else 0
        return tf

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
            header = ['Term'] + [f'TFIDF{i + 1}' for i in range(len(test_documents))]
            writer.writerow(header)

            # Data rows
            for term in self.terms:
                row = [term]
                tfidf_values = [self.compute_tfidf(doc)[term] for doc in test_documents]
                row.extend(tfidf_values)
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