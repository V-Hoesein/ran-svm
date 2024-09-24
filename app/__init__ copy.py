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


#* Fungsi untuk generate csv metriks tfidf
def generate_csv_manual():
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'dataset.csv'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'tfidf.csv'))
    
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
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'dataset.csv'))
    model_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'model.pkl'))
    result_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'tfidf.csv'))
    X_train_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'X_train.pkl'))
    
    # Mengambil label pada dataset csv
    data_csv = pd.read_csv(dataset_path)
    documents_sentiment = list(data_csv['label'])

    # Preprocssing
    text_cleaner = TextCleaner()
    
    comments = list(data_csv['comment'])
    raw_dataset = [text_cleaner.preprocess_text(raw) for raw in comments]
    
    # Calculating TFIDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(raw_dataset)
    
    joblib.dump(vectorizer, X_train_path)
    
    # Konversi label sentimen menjadi angka 0 untuk negatif dan 1 untuk positif
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(documents_sentiment)

    # Membuat Model Train SVM
    model = SVC(kernel='linear')

    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

train_model()

#* Fungsi untuk prediksi
def predict(test_text: str):
    dataset_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'dataset.csv'))
    X_train_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'X_train.pkl'))
    
    # data = pd.read_csv(dataset_path)
    # raw_data = list(data['comment'])

    # Preprocessing test text
    preprocessor = TextCleaner()
    # data_train = [preprocessor.preprocess_text(raw) for raw in raw_data]

    # Vectorization
    
    tfidf_model = joblib.load(X_train_path)
    tfidf_matrix = tfidf_model.transform([test_text])
    
    # Export to CSV
    terms = tfidf_model.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
    tfidf_df_sorted = tfidf_df.sort_index(axis=1)
    tfidf_df_sorted.to_csv('tfidf_matrix.csv', index=False)
    
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(X_train_path)

    # # Convert to DataFrame
    # terms = vectorizer.get_feature_names_out()
    # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)

    # # Save to CSV
    # tfidf_df_sorted = tfidf_df.sort_index(axis=1)
    # tfidf_df_sorted.to_csv('tfidf_from_lib.csv', index=False)

    # # Prepare data testing
    data_test = preprocessor.preprocess_text(test_text)

    # FIT data testing
    # tfidf_matrix_test = vectorizer.transform([data_test])
    
    # tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=terms)
    
    # # Hitung TF mentah
    # raw_counts = pd.Series(data_test.split()).value_counts()
    # tf_values = [raw_counts.get(term, 0) for term in terms]

    # # Tambahkan kolom TF mentah
    # tfidf_df_sorted_test = tfidf_df_test.sort_index(axis=1).T
    # tfidf_df_sorted_test['TF'] = tf_values  # Tambahkan kolom TF

    # # Menyesuaikan urutan kolom
    # tfidf_df_sorted_test = tfidf_df_sorted_test[['TF', 0]]  # Menggunakan indeks 0 untuk TFIDFN

    # # Simpan ke CSV
    # tfidf_df_sorted_test.columns = ['TF', 'TFIDFN']  # Sesuaikan nama kolom
    # tfidf_df_sorted_test.to_csv('tfidf_fit_from_lib.csv', index_label='Terms')
    
    
    #! SVM
    # model_path = os.path.realpath(os.path.join(os.path.dirname(__file__),  'static', 'uploads', 'model.pkl'))
    
    # model = joblib.load(model_path)
    
    # # Convert sparse matrix to dense array
    # tfidf_matrix_test_dense = tfidf_matrix_test.toarray()
    
    # prediction = model.predict(tfidf_matrix_test_dense)
    
    # return prediction


# generate_csv_manual()

# train_model()


# print(predict('Tidak sesuai sama yng d pesen, yang d pesen sadeor yng datang hymeys.... Sangat tidak sesuai sama yang d pesen....'))