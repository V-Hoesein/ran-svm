import os
from .preprocess import TextPreprocessor
from .tfidf import ManualTFIDF, TFIDFTest
from .svm import SVM
import pandas as pd
import numpy as np
import joblib  # Pastikan joblib diimpor

# Define file paths
dataset_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'dataset.csv'))
result_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads'))
model_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'model.pkl'))
tfidf_model_path = os.path.realpath(os.path.join(os.path.dirname(__name__), 'app', 'static', 'uploads', 'tfidf_model.pkl'))

def predict(test_documents: list):
    # Preprocessing instance
    preprocessor = TextPreprocessor()
   
    # Model instance
    svm_model = SVM()

    # Check if model and TF-IDF model already exist
    if os.path.exists(model_path) and os.path.exists(tfidf_model_path):
        # Load the trained SVM model and TF-IDF model
        svm_model.w, svm_model.b = joblib.load(model_path)
        
        # Gunakan load_model() untuk memuat TF-IDF model
        tfidf_model = ManualTFIDF([])
        tfidf_model.load_model(tfidf_model_path)  # Memastikan objek yang benar dimuat
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
        svm_model.fit(X_train, y_train)
        
        # Save trained model and TF-IDF model
        joblib.dump((svm_model.w, svm_model.b), model_path)
        tfidf_model.save_model(tfidf_model_path)  # Gunakan save_model() untuk menyimpan objek TFIDF
        print("Model and TF-IDF model trained and saved to", model_path, "and", tfidf_model_path)

    # Process test documents with TF-IDF
    tfidf_test = TFIDFTest(tfidf_model)  # Pass the TFIDF object directly
    tfidf_test.export_to_csv(test_documents, f'{result_path}/tfidf_test_result.csv')

    # Calculate TF-IDF matrix for test documents
    X_test = tfidf_test.get_tfidf_matrix(test_documents)

    # Perform predictions on test documents
    predictions = svm_model.predict(X_test)

    predictions = ["negatif" if p < 0 else "positif" for p in predictions]
    print("Predictions:", predictions)
    return predictions

# # Test documents to classify
# test_documents = [
#     "jelek kali",
# ]

# # Call the predict function with test documents
# res = predict(test_documents)

# print(res)
