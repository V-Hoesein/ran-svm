import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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

    def preprocess_from_csv(self, csv_file):
        # Membaca file CSV dan memproses kolom 'comment'
        data = pd.read_csv(csv_file)
        data['processed_comment'] = data['comment'].apply(self.preprocess)
        return data[['comment', 'processed_comment']]