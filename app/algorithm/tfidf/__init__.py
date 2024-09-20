import joblib
import math
import csv
import numpy as np

class TFIDFTest:
    def __init__(self, training_tfidf):
        print("Initializing TFIDFTest...")
        self.terms = training_tfidf.terms
        self.idf = training_tfidf.idf
        print("TFIDFTest initialized with terms and IDF.")

    def compute_tf(self, document):
        print("Computing TF for the document...")
        # Hitung TF untuk setiap term di dokumen
        tf = {}
        words = document.split()
        doc_len = len(words)
        for term in self.terms:
            tf[term] = words.count(term) / doc_len if doc_len > 0 else 0
        print("TF computed:", tf)
        return tf

    def compute_tfidf(self, test_document):
        print("Computing TF-IDF for the test document...")
        # Hitung TF-IDF untuk dokumen tes
        tf = self.compute_tf(test_document)
        tfidf = {}
        for term in self.terms:
            tfidf[term] = tf[term] * self.idf.get(term, 0)  # Ambil IDF dengan default 0 jika term tidak ada
        print("TF-IDF computed:", tfidf)
        return tfidf

    def get_tfidf_matrix(self, test_documents):
        print("Getting TF-IDF matrix for test documents...")
        # Mengembalikan matriks TF-IDF untuk semua dokumen tes
        tfidf_matrix = []
        for doc in test_documents:
            tfidf_values = self.compute_tfidf(doc)
            tfidf_matrix.append([tfidf_values.get(term, 0) for term in self.terms])  # Ambil nilai TF-IDF untuk setiap term
        print("TF-IDF matrix obtained:", tfidf_matrix)
        return np.array(tfidf_matrix)  # Kembalikan sebagai numpy array

    def export_to_csv(self, test_documents, filename):
        print(f"Exporting TF-IDF results to CSV: {filename}...")
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
                print(f"Written TF-IDF for term '{term}': {row}")

class ManualTFIDF:
    def __init__(self, documents):
        print("Initializing ManualTFIDF...")
        self.documents = documents
        self.terms = self.extract_terms()
        self.tf_matrix = self.compute_tf()
        self.df = self.compute_df()
        self.idf = self.compute_idf()
        self.tfidf_matrix = self.compute_tfidf()
        print("ManualTFIDF initialized.")

    def extract_terms(self):
        print("Extracting terms from documents...")
        # Ekstraksi semua terms unik dari seluruh dokumen
        terms = set()
        for doc in self.documents:
            terms.update(doc.split())
        terms_sorted = sorted(terms)
        print("Terms extracted:", terms_sorted)
        return terms_sorted

    def compute_tf(self):
        print("Computing TF for all documents...")
        # Hitung TF untuk setiap term di setiap dokumen
        tf_matrix = []
        for doc in self.documents:
            term_count = {}
            words = doc.split()
            doc_len = len(words)
            for term in self.terms:
                term_count[term] = words.count(term)  # Frekuensi mentah
            tf_matrix.append(term_count)
        print("TF matrix computed:", tf_matrix)
        return tf_matrix

    def compute_df(self):
        print("Computing DF (Document Frequency) for terms...")
        # Hitung DF (Document Frequency) untuk setiap term
        df = {term: 0 for term in self.terms}
        for term in self.terms:
            df[term] = sum(1 for doc in self.documents if term in doc.split())
        print("DF computed:", df)
        return df

    def compute_idf(self):
        print("Computing IDF based on DF...")
        # Hitung IDF berdasarkan DF
        N = len(self.documents)
        idf = {term: math.log(N / (df + 1)) for term, df in self.df.items()}
        print("IDF computed:", idf)
        return idf

    def compute_tfidf(self):
        print("Computing TF-IDF...")
        # Hitung TF-IDF dengan mengalikan TF dan IDF
        tfidf_matrix = []
        for tf_doc in self.tf_matrix:
            tfidf_doc = {}
            for term in tf_doc.keys():
                tfidf_doc[term] = (tf_doc[term] / len(self.documents)) * self.idf[term]  # TF-Normalized
            tfidf_matrix.append(tfidf_doc)
        print("TF-IDF matrix computed:", tfidf_matrix)
        return tfidf_matrix

    def compute_normalized_tf(self):
        print("Computing normalized TF...")
        # Hitung TF Normalisasi
        normalized_tf_matrix = []
        for tf_doc in self.tf_matrix:
            tf_norm_doc = {}
            max_tf = max(tf_doc.values()) if tf_doc else 1
            for term in tf_doc.keys():
                tf_norm_doc[term] = tf_doc[term] / max_tf if max_tf > 0 else 0
            normalized_tf_matrix.append(tf_norm_doc)
        print("Normalized TF computed.")
        return normalized_tf_matrix

    def export_to_csv(self, filename):
        print(f"Exporting results to CSV: {filename}...")
        normalized_tf = self.compute_normalized_tf()
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            header = ['Term']
            header += [f'TFD{i + 1}' for i in range(len(self.documents))]
            header += [f'TFNorm{i + 1}' for i in range(len(self.documents))]
            header.append('DF')
            header += [f'TFIDF{i + 1}' for i in range(len(self.documents))]
            writer.writerow(header)

            # Data rows
            for term in self.terms:
                row = [term]
                row += [self.tf_matrix[doc_idx][term] for doc_idx in range(len(self.documents))]
                row += [normalized_tf[doc_idx][term] for doc_idx in range(len(self.documents))]
                row.append(self.df[term])
                row += [self.tfidf_matrix[doc_idx][term] for doc_idx in range(len(self.documents))]
                writer.writerow(row)
                print(f"Written data for term '{term}': {row}")

    def get_tfidf_matrix(self):
        print("Getting TF-IDF matrix as numpy array...")
        # Mengembalikan matriks TF-IDF sebagai numpy array
        tfidf_array = np.array([[self.tfidf_matrix[i][term] for term in self.terms] for i in range(len(self.tfidf_matrix))])
        print("TF-IDF matrix obtained.")
        return tfidf_array
    
    def save_model(self, filepath):
        print(f"Saving model to {filepath}...")
        # Simpan model ke file .pkl
        model_data = {
            'terms': self.terms,
            'tf_matrix': self.tf_matrix,
            'df': self.df,
            'idf': self.idf,
            'tfidf_matrix': self.tfidf_matrix
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath):
        print(f"Loading model from {filepath}...")
        # Muat model dari file .pkl
        model_data = joblib.load(filepath)
        self.terms = model_data['terms']
        self.tf_matrix = model_data['tf_matrix']
        self.df = model_data['df']
        self.idf = model_data['idf']
        self.tfidf_matrix = model_data['tfidf_matrix']
        print(f"Model loaded from {filepath}.")
