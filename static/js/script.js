// DOM Elements
const reviewInput = document.getElementById('review-input');
const charCount = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const resultSection = document.getElementById('result-section');
const resetBtn = document.getElementById('reset-btn');
const loadingOverlay = document.getElementById('loading-overlay');

// Result elements
const sentimentIcon = document.getElementById('sentiment-icon');
const sentimentValue = document.getElementById('sentiment-value');
const confidenceValue = document.getElementById('confidence-value');
const confidenceFill = document.getElementById('confidence-fill');
const positiveCount = document.getElementById('positive-count');
const negativeCount = document.getElementById('negative-count');

// Character counter
reviewInput.addEventListener('input', () => {
    const length = reviewInput.value.length;
    charCount.textContent = length;

    // Enable/disable button based on input
    analyzeBtn.disabled = length === 0;
});

// Analyze button click handler
analyzeBtn.addEventListener('click', async () => {
    const reviewText = reviewInput.value.trim();

    if (!reviewText) {
        alert('Please enter a review to analyze.');
        return;
    }

    await analyzeSentiment(reviewText);
});

// Reset button click handler
resetBtn.addEventListener('click', () => {
    reviewInput.value = '';
    charCount.textContent = '0';
    resultSection.classList.add('hidden');
    analyzeBtn.disabled = true;
    reviewInput.focus();
});

// Allow Enter key to submit (with Shift+Enter for new line)
reviewInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (reviewInput.value.trim()) {
            analyzeBtn.click();
        }
    }
});

// Main sentiment analysis function
async function analyzeSentiment(reviewText) {
    try {
        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Make API request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: reviewText })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide loading overlay
        loadingOverlay.classList.add('hidden');

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error analyzing sentiment:', error);
        loadingOverlay.classList.add('hidden');
        alert('An error occurred while analyzing the review. Please try again.');
        analyzeBtn.disabled = false;
    }
}

// Display results in the UI
function displayResults(data) {
    const { sentiment, confidence, probabilities } = data;

    // Update sentiment display
    sentimentValue.textContent = sentiment;

    // Update icon and colors
    sentimentIcon.classList.remove('positive', 'negative', 'neutral');
    sentimentValue.classList.remove('positive', 'negative', 'neutral');

    if (sentiment === 'Positive') {
        sentimentIcon.classList.add('positive');
        sentimentValue.classList.add('positive');
    } else if (sentiment === 'Negative') {
        sentimentIcon.classList.add('negative');
        sentimentValue.classList.add('negative');
    } else {
        sentimentIcon.classList.add('neutral');
        sentimentValue.classList.add('neutral');
    }

    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${confidencePercent}%`;

    // Update score breakdown with probabilities (3 classes)
    const posPercent = Math.round((probabilities?.positive || 0) * 100);
    const negPercent = Math.round((probabilities?.negative || 0) * 100);
    const neuPercent = Math.round((probabilities?.neutral || 0) * 100);

    positiveCount.textContent = `${posPercent}%`;
    negativeCount.textContent = `${negPercent}%`;

    // Add neutral display if element exists
    const neutralCount = document.getElementById('neutral-count');
    if (neutralCount) {
        neutralCount.textContent = `${neuPercent}%`;
    }

    // Show result section with animation
    resultSection.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    // Re-enable analyze button
    analyzeBtn.disabled = false;
}

// Load and display model metrics
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) return;

        const metrics = await response.json();
        const metricsContent = document.getElementById('metrics-content');

        if (!metricsContent) return;

        const trainAcc = (metrics.train.accuracy * 100).toFixed(2);
        const testAcc = (metrics.test.accuracy * 100).toFixed(2);
        const testPrec = (metrics.test.precision * 100).toFixed(2);
        const testRec = (metrics.test.recall * 100).toFixed(2);
        const testF1 = (metrics.test.f1_score * 100).toFixed(2);

        metricsContent.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Training Accuracy</div>
                    <div class="metric-value">${trainAcc}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">${testAcc}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${testPrec}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${testRec}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value">${testF1}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Vocabulary Size</div>
                    <div class="metric-value">${metrics.vocabulary_size}</div>
                </div>
            </div>
            <div class="implementation-note">
                <strong>✨ Built from Scratch:</strong> No external NLP libraries used. 
                All components (preprocessing, BoW, Naive Bayes, metrics) implemented manually.
            </div>
        `;
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    reviewInput.focus();
    analyzeBtn.disabled = true;
    loadMetrics();
});
