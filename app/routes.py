import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import pandas as pd
from .algorithm import main
import logging

main_routes = Blueprint('main_routes', __name__)

# Constants
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
FILE_DATASET = os.path.join(UPLOAD_FOLDER, 'dataset.csv')

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to load dataset
def load_dataset():
    if os.path.exists(FILE_DATASET):
        return pd.read_csv(FILE_DATASET)
    return None

def clear_uploads_folder():
    print('delete files')
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if filename != 'dataset.csv' and os.path.isfile(file_path):
            os.remove(file_path)

# ** ROUTES ----- Functionalities ----- **
@main_routes.route('/')
def index():
    # df = load_dataset()
    # total_rows = df.shape[0] if df is not None else 0
    
    # keys = ['result', 'text', 'k', 'feature', 'panel', 'train_percentage', 'k2', 'feature2', 'result2', 'metrics']
    # context = {key: session.pop(key, None) for key in keys}
    
    # file_url = url_for('static', filename='uploads/dataset.csv') if df is not None else None
    
    # if not file_url:
    #     flash("Silahkan Upload Dataset!", 'warning')
    
    return render_template('index.html')


@main_routes.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        flash('No file part', 'error')
        return redirect(url_for('main_routes.index'))

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('main_routes.index'))

    if file and file.filename.endswith('.csv'):
        try:
            file.save(FILE_DATASET)
            flash('File successfully uploaded!', 'success')
        except Exception as e:
            flash(f'Failed to upload file: {e}', 'error')
    else:
        flash('File type not allowed. Please upload a .csv file.', 'error')

    return redirect(url_for('main_routes.index'))


@main_routes.route('/delete', methods=['POST'])
def delete_file():
    try:
        if os.path.exists(FILE_DATASET):
            os.remove(FILE_DATASET)
            flash('File dataset.csv berhasil dihapus.', 'success')
        else:
            flash('File dataset.csv tidak ditemukan.', 'error')
    except Exception as e:
        flash(f'Gagal menghapus file: {e}', 'error')
    
    return redirect(url_for('main_routes.index'))


@main_routes.route('/delete_all', methods=['POST'])
def delete_all_files():
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        
        if not files:
            flash('Tidak ada file yang ditemukan di folder uploads.', 'info')
        else:
            for file_name in files:
                os.remove(os.path.join(UPLOAD_FOLDER, file_name))
            flash('Semua file di folder uploads berhasil dihapus.', 'success')
    
    except Exception as e:
        flash(f'Gagal menghapus file: {e}', 'error')
    
    return redirect(url_for('main_routes.index'))


# ** ROUTES ----- ALGORITHM ----- **
@main_routes.route('/single_classification', methods=['POST'])
def single_classification():
    try:
        clear_uploads_folder()
        feature = int(request.form.get('feature'))
        k = int(request.form.get('k'))
        text = request.form.get('text')
        panel = request.form.get('panel')
        
        df = load_dataset()
        if df is None:
            flash('Dataset not found!', 'error')
            return redirect(url_for('main_routes.index'))
        
        text_dict = [{'text': text, 'type': ''}]
        list_of_dict = df.to_dict(orient='records')
        result = main(feature, k, list_of_dict, text_dict)
        
        session.update({'result': result, 'text': text, 'k': k, 'feature': feature, 'panel': panel})
        flash('Classification successful!', 'success')
        return redirect(url_for('main_routes.index'))
    
    except ValueError as ve:
        flash(f'Value error: {ve}', 'error')
        logging.error(f"Value error: {ve}")
        return redirect(url_for('main_routes.index'))
    except Exception as e:
        flash(f'An unexpected error occurred: {e}', 'error')
        logging.error(f"An unexpected error occurred: {e}")
        return redirect(url_for('main_routes.index'))

@main_routes.route('/multi_classification', methods=['POST'])
def multi_classification():
    try:
        clear_uploads_folder()
        feature2 = int(request.form.get('feature2'))
        k2 = int(request.form.get('k2'))
        train_percentage = int(request.form.get('train'))  # Train percentage (0-100)
        panel = request.form.get('panel')

        df = load_dataset()
        if df is None:
            raise ValueError("Dataset tidak ditemukan!")
        
        num_rows = df.shape[0]
        train_size = int(num_rows * (train_percentage / 100))

        data_train = df.iloc[:train_size].to_dict(orient='records')  # First N% of the data
        data_test = df.iloc[train_size:].to_dict(orient='records')  # Remaining (100-N)% of the data

        predicts, metrics = main(feature2, k2, data_train, data_test, multi=True)
        
        print(predicts)
        print(metrics)
        
        session.update({
            'result2': predicts,
            'metrics': metrics,
            'k2': k2,
            'feature2': feature2,
            'panel': panel,
            'train_percentage': train_percentage
        })
        return redirect(url_for('main_routes.index'))
    
    except Exception as e:
        flash(f'Error during classification: {e}', 'error')
        return redirect(url_for('main_routes.index'))
