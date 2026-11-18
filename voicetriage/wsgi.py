# Entry point for gunicorn: `gunicorn wsgi:app`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from webapp.app import app  # Flask instance in webapp/app.py
