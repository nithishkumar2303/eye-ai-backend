from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging

from src.models import db
from src.auth_routes import auth_bp
from src. prediction_routes import prediction_bp

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory"""
    app = Flask(__name__)
    return app
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///eye_ai.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": os.getenv('ALLOWED_ORIGINS', '*').split(',')}})
    
    # Register blueprints
    app. register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("✅ Database initialized")
    
    # Health check route
    @app.route('/health', methods=['GET'])
    def health():
        return {'status': 'healthy'}, 200
    
    # Root route
    @app.route('/', methods=['GET'])
    def root():
        return {
            'message': 'Eye-AI Backend API',
            'version': '1.0.0',
            'endpoints': {
                'auth': '/auth',
                'predictions': '/predictions',
                'health': '/health'
            }
        }, 200
    
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)