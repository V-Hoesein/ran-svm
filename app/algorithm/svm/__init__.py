import numpy as np
import joblib
import csv

class SVM:
    def __init__(self, learning_rate=0.01, regularization_strength=0.01, n_iters=1000):
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
        y_ = np.where(y <= 0, -1, 1)

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
        return np.sign(linear_output)

    def export_weights_to_csv(self, filename):
        if self.terms is None or self.w is None:
            print("Model not trained or terms not set.")
            return

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
        joblib.dump((self.w, self.b), file_path)
