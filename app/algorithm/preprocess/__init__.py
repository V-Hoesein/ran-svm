import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Sastrawi Stemmer dan StopWordRemover
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        print("TextPreprocessor initialized.")

    def clean_text(self, text):
        print("Cleaning text...")
        # Menghapus karakter khusus dan angka, serta mengubah menjadi huruf kecil
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter non-alphabetic
        text = text.lower()  # Mengubah teks menjadi huruf kecil
        print("Text cleaned:", text)
        return text

    def preprocess(self, text):
        print("Starting preprocessing...")
        # Melakukan text cleansing, menghapus stopwords dan stemming
        text = self.clean_text(text)  # Membersihkan teks
        text = self.stopword_remover.remove(text)  # Menghapus stopwords
        print("Stopwords removed:", text)
        text = self.stemmer.stem(text)  # Melakukan stemming
        print("Text after stemming:", text)
        return text

    def preprocess_from_csv(self, csv_file):
        print(f"Reading CSV file: {csv_file}")
        # Membaca file CSV dan memproses kolom 'comment'
        data = pd.read_csv(csv_file)
        print("Processing comments...")
        data['processed_comment'] = data['comment'].apply(self.preprocess)
        print("Processing complete.")
        return data[['comment', 'processed_comment']]
