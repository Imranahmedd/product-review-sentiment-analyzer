"""
NLP Core Module - Built from Scratch
No external NLP libraries used (no spaCy, NLTK, scikit-learn for NLP)
All components implemented using fundamental Python concepts

Components:
1. TextPreprocessor - Tokenization, cleaning, stopword removal
2. BagOfWordsVectorizer - Text to numerical vector conversion
3. NaiveBayesClassifier - Multinomial Naive Bayes implementation
4. MetricsCalculator - Evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import re
import json
import math
from collections import defaultdict, Counter


class TextPreprocessor:
    """
    Handles all text preprocessing tasks from scratch.
    Enhanced with negation handling.
    """
    
    # Common English stopwords (manually defined)
    # REMOVED negation words from stopwords to preserve sentiment context
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
        'will', 'just', 'should', 'now'
    }
    
    # Negation words - built from scratch list
    NEGATIONS = {
        'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
        'none', 'nor', "don't", "doesn't", "didn't", "won't", "wouldn't",
        "shouldn't", "can't", "cannot", "couldn't", "mustn't", "mightn't",
        "shan't", "hasn't", "haven't", "hadn't", "isn't", "aren't", "wasn't",
        "weren't", 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt',
        'cant', 'couldnt', 'mustnt', 'mightnt', 'shant', 'hasnt', 'havent',
        'hadnt', 'isnt', 'arent', 'wasnt', 'werent'
    }
    
    def __init__(self, remove_stopwords=True, min_word_length=2, handle_negation=True):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            min_word_length (int): Minimum length for a word to be kept
            handle_negation (bool): Whether to handle negations
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.handle_negation = handle_negation
    
    def tokenize(self, text):
        """
        Convert text into tokens (words) with negation handling.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Replace punctuation with spaces (except apostrophes in contractions)
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Split into words
        tokens = text.split()
        
        # Handle negations BEFORE removing stopwords
        if self.handle_negation:
            tokens = self._handle_negations(tokens)
        
        # Remove very short words
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Remove stopwords if enabled (but keep negated words)
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.STOPWORDS or token.startswith('NOT_')]
        
        return tokens
    
    def _handle_negations(self, tokens):
        """
        Handle negations by marking words following negation words.
        Built from scratch without external libraries.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with negations handled
        """
        result = []
        negate = False
        negation_window = 3  # Mark next 3 words after negation
        words_since_negation = 0
        
        for token in tokens:
            # Check if current token is a negation word
            if token in self.NEGATIONS:
                negate = True
                words_since_negation = 0
                # Don't add the negation word itself to preserve "NOT_" marking
                continue
            
            # If we're in negation mode, mark the word
            if negate and words_since_negation < negation_window:
                # Skip punctuation and very short words
                if len(token) >= 2:
                    result.append(f'NOT_{token}')
                    words_since_negation += 1
                else:
                    result.append(token)
            else:
                result.append(token)
                negate = False  # Reset negation after window
        
        return result
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Preprocessed tokens
        """
        return self.tokenize(text)


class BagOfWordsVectorizer:
    """
    Converts text documents into numerical vectors using Bag-of-Words approach.
    Enhanced with N-gram support (unigrams + bigrams).
    Built from scratch without sklearn.
    """
    
    def __init__(self, max_features=None, min_df=1, ngram_range=(1, 2)):
        """
        Initialize the vectorizer.
        
        Args:
            max_features (int): Maximum number of features (most frequent words)
            min_df (int): Minimum document frequency for a word to be included
            ngram_range (tuple): Range of n-grams to extract (min_n, max_n)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary = {}  # word -> index mapping
        self.word_counts = Counter()  # word -> document count
        self.preprocessor = TextPreprocessor()
    
    def _generate_ngrams(self, tokens, n):
        """
        Generate n-grams from tokens.
        Built from scratch without external libraries.
        
        Args:
            tokens (list): List of tokens
            n (int): N-gram size
            
        Returns:
            list: List of n-grams (as strings)
        """
        if n == 1:
            return tokens
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            # Join n consecutive tokens with underscore
            ngram = '_'.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def _extract_features(self, tokens):
        """
        Extract all n-grams based on ngram_range.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: All n-grams
        """
        all_ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            ngrams = self._generate_ngrams(tokens, n)
            all_ngrams.extend(ngrams)
        
        return all_ngrams
    
    def fit(self, documents):
        """
        Build vocabulary from training documents.
        
        Args:
            documents (list): List of text documents
        """
        # Count word/ngram occurrences across documents
        doc_feature_sets = []
        
        for doc in documents:
            tokens = self.preprocessor.preprocess(doc)
            features = self._extract_features(tokens)
            doc_feature_sets.append(set(features))
        
        # Count in how many documents each feature appears
        for feature_set in doc_feature_sets:
            for feature in feature_set:
                self.word_counts[feature] += 1
        
        # Filter by minimum document frequency
        filtered_features = [
            feature for feature, count in self.word_counts.items() 
            if count >= self.min_df
        ]
        
        # Select top features if max_features is set
        if self.max_features and len(filtered_features) > self.max_features:
            # Sort by frequency and take top N
            sorted_features = sorted(
                filtered_features, 
                key=lambda w: self.word_counts[w], 
                reverse=True
            )
            filtered_features = sorted_features[:self.max_features]
        
        # Create vocabulary (feature -> index mapping)
        self.vocabulary = {feature: idx for idx, feature in enumerate(sorted(filtered_features))}
    
    def transform(self, documents):
        """
        Transform documents into BoW vectors.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of BoW vectors (each vector is a list of feature counts)
        """
        vectors = []
        
        for doc in documents:
            tokens = self.preprocessor.preprocess(doc)
            features = self._extract_features(tokens)
            
            # Initialize vector with zeros
            vector = [0] * len(self.vocabulary)
            
            # Count feature occurrences
            for feature in features:
                if feature in self.vocabulary:
                    idx = self.vocabulary[feature]
                    vector[idx] += 1
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, documents):
        """
        Fit vocabulary and transform documents in one step.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of BoW vectors
        """
        self.fit(documents)
        return self.transform(documents)
    
    def get_feature_names(self):
        """
        Get list of feature names (words/ngrams) in vocabulary order.
        
        Returns:
            list: Ordered list of features
        """
        return sorted(self.vocabulary.keys(), key=lambda w: self.vocabulary[w])
    
    def save(self, filepath):
        """Save vocabulary to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocabulary': self.vocabulary,
                'word_counts': dict(self.word_counts),
                'max_features': self.max_features,
                'min_df': self.min_df,
                'ngram_range': self.ngram_range
            }, f, indent=2)
    
    def load(self, filepath):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocabulary = data['vocabulary']
            self.word_counts = Counter(data['word_counts'])
            self.max_features = data['max_features']
            self.min_df = data['min_df']
            self.ngram_range = tuple(data.get('ngram_range', (1, 1)))


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier built from scratch.
    Uses Laplace smoothing to handle unseen words.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier.
        
        Args:
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.classes = []
        self.class_priors = {}  # P(class)
        self.feature_probs = {}  # P(word|class)
        self.class_word_counts = {}  # Total words per class
        self.vocabulary_size = 0
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X (list): List of BoW vectors
            y (list): List of labels (0 or 1)
        """
        # Get unique classes
        self.classes = sorted(list(set(y)))
        n_samples = len(y)
        self.vocabulary_size = len(X[0]) if X else 0
        
        # Calculate class priors: P(class) = count(class) / total_samples
        class_counts = Counter(y)
        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples
        
        # Calculate feature probabilities for each class
        for cls in self.classes:
            # Get all vectors for this class
            class_vectors = [X[i] for i in range(len(X)) if y[i] == cls]
            
            # Sum word counts across all documents in this class
            word_counts = [0] * self.vocabulary_size
            for vector in class_vectors:
                for idx, count in enumerate(vector):
                    word_counts[idx] += count
            
            # Total words in this class
            total_words = sum(word_counts)
            self.class_word_counts[cls] = total_words
            
            # Calculate P(word|class) with Laplace smoothing
            # P(word|class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
            self.feature_probs[cls] = []
            for count in word_counts:
                prob = (count + self.alpha) / (total_words + self.alpha * self.vocabulary_size)
                self.feature_probs[cls].append(prob)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Args:
            X (list): List of BoW vectors
            
        Returns:
            list: List of probability dictionaries for each sample
        """
        predictions = []
        
        for vector in X:
            # Calculate log probabilities for numerical stability
            log_probs = {}
            
            for cls in self.classes:
                # Start with log of class prior
                log_prob = math.log(self.class_priors[cls])
                
                # Add log probabilities for each word
                for idx, count in enumerate(vector):
                    if count > 0:
                        # Multiply probability by count (add log(prob) * count)
                        log_prob += count * math.log(self.feature_probs[cls][idx])
                
                log_probs[cls] = log_prob
            
            # Convert log probabilities back to probabilities
            # Use log-sum-exp trick for numerical stability
            max_log_prob = max(log_probs.values())
            exp_probs = {cls: math.exp(lp - max_log_prob) for cls, lp in log_probs.items()}
            total = sum(exp_probs.values())
            probs = {cls: p / total for cls, p in exp_probs.items()}
            
            predictions.append(probs)
        
        return predictions
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X (list): List of BoW vectors
            
        Returns:
            list: Predicted class labels
        """
        proba = self.predict_proba(X)
        return [max(p.items(), key=lambda x: x[1])[0] for p in proba]
    
    def save(self, filepath):
        """Save model to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'alpha': self.alpha,
                'classes': self.classes,
                'class_priors': self.class_priors,
                'feature_probs': self.feature_probs,
                'class_word_counts': self.class_word_counts,
                'vocabulary_size': self.vocabulary_size
            }, f, indent=2)
    
    def load(self, filepath):
        """Load model from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.alpha = data['alpha']
            self.classes = data['classes']
            self.class_priors = {int(k) if k.isdigit() else k: v for k, v in data['class_priors'].items()}
            self.feature_probs = {int(k) if k.isdigit() else k: v for k, v in data['feature_probs'].items()}
            self.class_word_counts = {int(k) if k.isdigit() else k: v for k, v in data['class_word_counts'].items()}
            self.vocabulary_size = data['vocabulary_size']


class MetricsCalculator:
    """
    Calculate evaluation metrics from scratch.
    """
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy score."""
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true) if y_true else 0.0
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, classes):
        """
        Calculate confusion matrix.
        
        Returns:
            dict: Confusion matrix as nested dict {true_class: {pred_class: count}}
        """
        matrix = {cls: {cls2: 0 for cls2 in classes} for cls in classes}
        
        for true, pred in zip(y_true, y_pred):
            matrix[true][pred] += 1
        
        return matrix
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred, positive_class=1):
        """
        Calculate precision, recall, and F1 score for binary classification.
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            positive_class: The class to consider as positive
            
        Returns:
            tuple: (precision, recall, f1)
        """
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == positive_class and pred == positive_class)
        fp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true != positive_class and pred == positive_class)
        fn = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == positive_class and pred != positive_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    @staticmethod
    def classification_report(y_true, y_pred, classes):
        """
        Generate a complete classification report.
        
        Returns:
            dict: Report with metrics for each class
        """
        report = {}
        
        for cls in classes:
            precision, recall, f1 = MetricsCalculator.precision_recall_f1(
                y_true, y_pred, positive_class=cls
            )
            support = sum(1 for y in y_true if y == cls)
            
            report[cls] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
        
        # Overall accuracy
        report['accuracy'] = MetricsCalculator.accuracy(y_true, y_pred)
        
        return report
