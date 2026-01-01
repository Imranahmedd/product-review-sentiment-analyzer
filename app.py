from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import re
import logging
from datetime import datetime

# Import our custom NLP modules (built from scratch)
from nlp_core import BagOfWordsVectorizer, NaiveBayesClassifier

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MAX_INPUT_LENGTH = 5000  # Maximum characters allowed
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for definitive prediction

# Global variables for model and vectorizer
vectorizer = None
classifier = None
metrics = None


def load_model():
    """Load the trained model and vocabulary on startup."""
    global vectorizer, classifier, metrics
    
    print("Loading trained model and vocabulary...")
    
    try:
        # Load vectorizer (Bag-of-Words)
        vectorizer = BagOfWordsVectorizer()
        vectorizer.load('models/vocabulary.json')
        print(f"✓ Vocabulary loaded: {len(vectorizer.vocabulary)} words")
        
        # Load classifier (Naive Bayes)
        classifier = NaiveBayesClassifier()
        classifier.load('models/naive_bayes_model.json')
        print(f"✓ Model loaded: {classifier.classes}")
        
        # Load metrics
        if os.path.exists('models/metrics.json'):
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
            print(f"✓ Metrics loaded")
        
        print("Model ready for predictions!\n")
        
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. Please run 'python train_model.py' first.")
        print(f"Details: {e}")
        raise


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for sentiment prediction using trained Naive Bayes model.
    Expects JSON: {"review": "text to analyze"}
    Returns: {"sentiment": "Positive/Negative/Uncertain", "confidence": 0.0-1.0, ...}
    """
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            logger.warning("Prediction request missing review field")
            return jsonify({'error': 'No review text provided'}), 400
        
        review_text = data['review'].strip()
        
        # Input validation
        if not review_text:
            return jsonify({'error': 'Review text cannot be empty'}), 400
        
        if len(review_text) > MAX_INPUT_LENGTH:
            logger.warning(f"Input too long: {len(review_text)} characters")
            return jsonify({'error': f'Review text too long. Maximum {MAX_INPUT_LENGTH} characters allowed.'}), 400
        
        # Sanitize input (remove HTML tags, scripts)
        review_text = _sanitize_input(review_text)
        
        # Transform review to BoW vector
        review_vector = vectorizer.transform([review_text])
        
        # Get prediction
        prediction = classifier.predict(review_vector)[0]
        
        # Get probability distribution
        proba = classifier.predict_proba(review_vector)[0]
        
        # Map prediction to sentiment (0=Negative, 1=Positive, 2=Neutral)
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        confidence = proba.get(prediction, 0.0)
        
        # Check confidence threshold (optional - for very low confidence)
        is_uncertain = confidence < MIN_CONFIDENCE_THRESHOLD
        if is_uncertain:
            logger.info(f"Low confidence prediction: {confidence:.4f}")
        
        # Get word counts for display (optional)
        tokens = vectorizer.preprocessor.preprocess(review_text)
        word_count = len(tokens)
        
        result = {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'prediction_class': int(prediction),
            'is_uncertain': is_uncertain,
            'probabilities': {
                'negative': round(proba.get(0, 0.0), 4),
                'positive': round(proba.get(1, 0.0), 4),
                'neutral': round(proba.get(2, 0.0), 4)
            },
            'word_count': word_count,
            'vocabulary_size': len(vectorizer.vocabulary)
        }
        
        # Log prediction
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction'}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    API endpoint to get model performance metrics.
    Returns training and test metrics.
    """
    try:
        if metrics is None:
            return jsonify({'error': 'Metrics not available'}), 404
        
        return jsonify(metrics), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    API endpoint to get model information.
    """
    try:
        info = {
            'model_type': 'Naive Bayes (Multinomial)',
            'vocabulary_size': len(vectorizer.vocabulary),
            'classes': classifier.classes,
            'implementation': 'Built from scratch (no external NLP libraries)',
            'features': 'Bag-of-Words (BoW)',
            'preprocessing': [
                'Lowercasing',
                'Punctuation removal',
                'Tokenization',
                'Stopword removal'
            ]
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _sanitize_input(text):
    """
    Sanitize user input to prevent XSS and injection attacks.
    Built from scratch without external libraries.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script tags and content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove potential SQL injection patterns (basic)
    text = re.sub(r'(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b)', '', text, flags=re.IGNORECASE)
    
    return text


if __name__ == '__main__':
    # Load model before starting server
    load_model()
    
    # Get environment configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    logger.info(f"Starting server on {host}:{port} (debug={debug_mode})")
    
    # Start Flask server (debug=False for production security)
    app.run(debug=debug_mode, host=host, port=port)
