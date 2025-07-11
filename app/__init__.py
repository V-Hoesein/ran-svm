from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'svm_applications'

    # Register blueprints or routes
    with app.app_context():
        from .routes import main_routes
        app.register_blueprint(main_routes)

    return app
