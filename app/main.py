"""Flask application factory."""
import os
import logging
from flask import Flask
from app.config import LOG_PATH, HOST, PORT, DEBUG
from app.database import Database
from app.models.model_manager import ModelManager
from app.models.drift_detector import DriftDetector
from app.pipeline.preprocessor import Preprocessor
from app.pipeline.rule_engine import RuleEngine
from app.pipeline.postprocessor import Postprocessor
from app.api.routes import init_api
from app.api.simulation import init_simulation
from app.dashboard.routes import init_dashboard


def create_app():
    """Create and configure the Flask application."""
    # Logging
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Fraud Detection System")

    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    app.config['SECRET_KEY'] = 'fyp-fraud-detection-2026'

    # Initialize components
    logger.info("Initializing database...")
    db = Database()

    logger.info("Loading models...")
    model_manager = ModelManager()

    logger.info("Initializing pipeline...")
    preprocessor = Preprocessor(db=db)
    rule_engine = RuleEngine()
    postprocessor = Postprocessor()
    drift_detector = DriftDetector()

    # Make 3-Model Staged Study constants available to all templates
    from app.config import (
        MODEL_CATEGORIES, MODEL_CATEGORY_LABELS, MODEL_DESCRIPTIONS,
        MODEL_F1_LABELS, MODEL_INTERNAL_TO_DISPLAY,
        STAGED_STUDY_TABLE_A, STAGED_STUDY_TABLE_B,
    )
    @app.context_processor
    def inject_staged_study_constants():
        return {
            'MODEL_CATEGORIES': MODEL_CATEGORIES,
            'MODEL_CATEGORY_LABELS': MODEL_CATEGORY_LABELS,
            'MODEL_DESCRIPTIONS': MODEL_DESCRIPTIONS,
            'MODEL_F1_LABELS': MODEL_F1_LABELS,
            'MODEL_INTERNAL_TO_DISPLAY': MODEL_INTERNAL_TO_DISPLAY,
            'STAGED_STUDY_TABLE_A': STAGED_STUDY_TABLE_A,
            'STAGED_STUDY_TABLE_B': STAGED_STUDY_TABLE_B,
        }

    # Register routes
    init_api(app, model_manager, preprocessor, rule_engine, postprocessor, drift_detector, db)
    init_simulation(app, model_manager, preprocessor, rule_engine, postprocessor, db)
    init_dashboard(app, model_manager, db)

    logger.info("System ready. Active model: %s", model_manager.active_model_name)
    return app, db, model_manager
