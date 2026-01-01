"""
Training Script for Sentiment Analysis Model
Trains a Naive Bayes classifier from scratch on product review data

This script:
1. Loads the dataset
2. Splits into train/test sets
3. Builds Bag-of-Words vocabulary
4. Trains Naive Bayes classifier
5. Evaluates performance
6. Saves model and vocabulary
"""

import csv
import random
import json
import os
from nlp_core import (
    BagOfWordsVectorizer,
    NaiveBayesClassifier,
    MetricsCalculator
)


def load_dataset(filepath):
    """
    Load dataset from CSV file.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        tuple: (reviews, labels)
    """
    reviews = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row['review'])
            labels.append(int(row['sentiment']))
    
    return reviews, labels


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (list): Features
        y (list): Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Set random seed
    random.seed(random_state)
    
    # Create indices and shuffle
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculate split point
    split_idx = int(len(X) * (1 - test_size))
    
    # Split indices
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split data
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


def print_metrics(y_true, y_pred, dataset_name):
    """Print evaluation metrics for a dataset."""
    
    # Get unique classes from the data
    classes = sorted(list(set(y_true)))
    
    # Calculate metrics
    accuracy = MetricsCalculator.accuracy(y_true, y_pred)
    
    # For multi-class, calculate weighted average metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_support = len(y_true)
    
    for cls in classes:
        precision, recall, f1 = MetricsCalculator.precision_recall_f1(y_true, y_pred, positive_class=cls)
        support = sum(1 for y in y_true if y == cls)
        weight = support / total_support
        total_precision += precision * weight
        total_recall += recall * weight
        total_f1 += f1 * weight
    
    cm = MetricsCalculator.confusion_matrix(y_true, y_pred, classes=classes)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {total_precision:.4f}")
    print(f"Recall:    {total_recall:.4f}")
    print(f"F1-Score:  {total_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    
    # Determine class labels for display
    if len(classes) == 2:
        class_labels = ["Neg", "Pos"]
    elif len(classes) == 3:
        class_labels = ["Neg", "Pos", "Neu"]
    else:
        class_labels = [f"C{c}" for c in classes]
    
    # Print header
    print(f"{'':16}Predicted")
    header = "              " + "  ".join(f"{label:>5}" for label in class_labels)
    print(header)
    
    # Print matrix rows
    for i, true_class in enumerate(classes):
        row_label = f"Actual  {class_labels[i]}"
        row_values = "  ".join(f"{cm[true_class][pred_class]:>5}" for pred_class in classes)
        print(f"{row_label:14}{row_values}")
    
    return {
        'accuracy': accuracy,
        'precision': total_precision,
        'recall': total_recall,
        'f1_score': total_f1,
        'confusion_matrix': cm
    }


def main():
    """Main training function."""
    print("="*60)
    print("Product Review Sentiment Analysis - Model Training")
    print("="*60)
    
    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    reviews, labels = load_dataset('dataset/reviews.csv')
    print(f"   ✓ Loaded {len(reviews)} reviews")
    print(f"   ✓ Positive reviews: {sum(labels)}")
    print(f"   ✓ Negative reviews: {len(labels) - sum(labels)}")
    
    # 2. Split dataset
    print("\n[2/6] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.2, random_state=42
    )
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Testing samples:  {len(X_test)}")
    
    # 3. Build Bag-of-Words vocabulary
    print("\n[3/6] Building Bag-of-Words vocabulary...")
    vectorizer = BagOfWordsVectorizer(max_features=1000, min_df=2)
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"   ✓ Feature vectors created")
    
    # Show some vocabulary examples
    vocab_words = vectorizer.get_feature_names()
    print(f"   ✓ Sample words: {', '.join(vocab_words[:10])}...")
    
    # 4. Train Naive Bayes classifier
    print("\n[4/6] Training Naive Bayes classifier...")
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train_vectors, y_train)
    print(f"   ✓ Model trained successfully")
    print(f"   ✓ Classes: {classifier.classes}")
    
    # 5. Evaluate on training set
    print("\n[5/6] Evaluating model...")
    y_train_pred = classifier.predict(X_train_vectors)
    train_metrics = print_metrics(y_train, y_train_pred, "Training Set")
    
    # Evaluate on test set
    y_test_pred = classifier.predict(X_test_vectors)
    test_metrics = print_metrics(y_test, y_test_pred, "Test Set")
    
    # 6. Save model and vocabulary
    print("\n[6/6] Saving model and vocabulary...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    classifier.save('models/naive_bayes_model.json')
    vectorizer.save('models/vocabulary.json')
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'vocabulary_size': len(vectorizer.vocabulary),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✓ Model saved to: models/naive_bayes_model.json")
    print(f"   ✓ Vocabulary saved to: models/vocabulary.json")
    print(f"   ✓ Metrics saved to: models/metrics.json")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Test with example reviews
    print("\n" + "="*60)
    print("Testing with Example Reviews")
    print("="*60)
    
    test_reviews = [
        "This product is absolutely amazing! Love it!",
        "Terrible quality. Waste of money. Very disappointed.",
        "Great value for money. Works perfectly!",
        "Broke after two days. Poor quality."
    ]
    
    for review in test_reviews:
        vector = vectorizer.transform([review])
        prediction = classifier.predict(vector)[0]
        proba = classifier.predict_proba(vector)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = proba[prediction]
        
        print(f"\nReview: \"{review}\"")
        print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
