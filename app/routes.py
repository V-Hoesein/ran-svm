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
    return render_template('index.html')
