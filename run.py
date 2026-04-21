"""Entry point: python run.py"""
from app.main import create_app
from app.config import HOST, PORT, DEBUG

app, db, model_manager = create_app()

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
