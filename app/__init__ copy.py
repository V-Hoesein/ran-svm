import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Contoh data latih
train_docs = [
    "Saya suka komputer",
    "Komputer dan laptop bagus",
    "Saya belajar dengan laptop dan komputer"
]

# Contoh data uji (teks baru)
new_text = "Suka komputer dan belajar"

# Membuat dan melatih TF-IDF Vectorizer menggunakan data latih
vectorizer = TfidfVectorizer()

# Melakukan fit pada data latih untuk mendapatkan kosakata dan IDF
X_train = vectorizer.fit_transform(train_docs)

# Menampilkan kosakata yang dipelajari oleh vectorizer
print("Kosakata (Terms):")
print(vectorizer.get_feature_names_out())

# Menghitung nilai IDF untuk setiap kata
idf_values = vectorizer.idf_
print("\nNilai IDF untuk setiap term:")
idf_df = pd.DataFrame({'Terms': vectorizer.get_feature_names_out(), 'IDF': idf_values})
print(idf_df)

# Melakukan transformasi pada teks baru menggunakan vectorizer yang sudah dilatih
X_new_text = vectorizer.transform([new_text])

# Mengambil nilai TF-IDF dari teks baru
print("\nTF-IDF untuk teks baru:")
tfidf_new_text = pd.DataFrame(X_new_text.T.toarray(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])
print(tfidf_new_text)

# Mengekspor hasil TF-IDF dari teks baru ke file CSV
tfidf_new_text.to_csv('tfidf_new_text.csv', index=True)
print("\nNilai TF-IDF teks baru diekspor ke tfidf_new_text.csv")
