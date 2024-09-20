import numpy as np
import joblib  # Pastikan untuk menginstal joblib dengan pip install joblib

class SVM:
    def __init__(self, learning_rate=0.01, regularization_strength=0.01, n_iters=1000):
        print("Initializing SVM with learning rate:", learning_rate, 
              "regularization strength:", regularization_strength, 
              "and number of iterations:", n_iters)
        self.lr = learning_rate
        self.reg_strength = regularization_strength
        self.n_iters = n_iters
        self.w = None  # Bobot
        self.b = None  # Bias

    def fit(self, X, y):
        print("Fitting the SVM model...")
        n_samples, n_features = X.shape
        # Inisialisasi bobot dan bias
        self.w = np.zeros(n_features)
        self.b = 0

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
        print("Model fitting completed.")

    def predict(self, X):
        print("Making predictions...")
        # Menghitung prediksi
        linear_output = np.dot(X, self.w) + self.b
        predictions = np.sign(linear_output)
        print("Predictions made:", predictions)
        return predictions

    def save_model(self, file_path):
        print(f"Saving model to {file_path}...")
        # Menyimpan model ke file .pkl
        joblib.dump((self.w, self.b), file_path)
        print("Model saved successfully.")
