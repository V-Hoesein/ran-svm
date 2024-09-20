import os
from preprocess import TextPreprocessor
from tfidf import ManualTFIDF, TFIDFTest
from svm import SVM
import pandas as pd
import numpy as np
import joblib  # Pastikan joblib diimpor

dataset_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app','static','uploads', 'dataset.csv'))
result_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app','static','uploads'))
model_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app','static','uploads', 'model.pkl'))

# Preprocessing
preprocessor = TextPreprocessor()
processed_data = preprocessor.preprocess_from_csv(dataset_path)
preprocessed_text = list(processed_data['processed_comment'])

# TFIDF build Model
data = pd.read_csv(dataset_path)
tfidf_model = ManualTFIDF(preprocessed_text)
tfidf_model.export_to_csv(f'{result_path}/tfidf_result.csv')
y_tfidf = list(data['label'])

# TFIDF Test apply Model
test_documents = [
    "jelek kali",
    "bagus banget aku suka"
]

tfidf_test = TFIDFTest(tfidf_model)
tfidf_test.export_to_csv(test_documents, f'{result_path}/tfidf_test_result.csv')

# SVM
X_train = tfidf_model.get_tfidf_matrix()  # Mendapatkan matriks TF-IDF
y_train = np.array(y_tfidf)  # Mengubah label menjadi array numpy
svm_model = SVM()

# Cek apakah model sudah ada
if os.path.exists(model_path):
    # Jika model sudah ada, muat model
    svm_model.w, svm_model.b = joblib.load(model_path)
    print("Model loaded from", model_path)
else:
    # Jika model belum ada, latih model
    svm_model.fit(X_train, y_train)
    svm_model.save_model(model_path)
    print("Model trained and saved to", model_path)

# Hitung TF-IDF untuk dokumen tes
X_test = tfidf_test.get_tfidf_matrix(test_documents)  # Pass test_documents sebagai argumen

# Lakukan prediksi pada dokumen tes
predictions = svm_model.predict(X_test)

print("Actuals : ", y_train)
print("Predictions:", predictions)
