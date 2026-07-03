import os
import sys
from flask import Flask, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


from config import Config
from utils.db import init_chat_db
from routes.chat import chat_bp
from routes.upload import upload_bp
from routes.appointments import appointments_bp
from routes.capabilities import capabilities_bp

def create_app():
    # Enforce SECRET_KEY in production
    if os.environ.get('FLASK_ENV') == 'production' and not os.environ.get('SECRET_KEY'):
        raise ValueError("No SECRET_KEY set for Flask application. This is required in production.")

    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

    # Initialize DB
    init_chat_db()

    # Register blueprints
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(appointments_bp, url_prefix='/api')
    app.register_blueprint(capabilities_bp, url_prefix='/api')

    # Rate limits per blueprint/route
    limiter.limit("30 per minute")(chat_bp)
    limiter.limit("10 per minute")(upload_bp)
    
    @app.route('/')
    def index():
        return render_template('index.html')

    return app

app = create_app()

if __name__ == '__main__':
    print("🚀 Starting ClinIQ Gateway (Port 5000)...")
    print("⚠️ Note for production: Use gunicorn with gevent workers, e.g.:")
    print("   gunicorn -w 4 -k gevent app:app -b 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
