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
        const metricsCharts = document.getElementById('metrics-charts');

        if (!metricsContent || !metricsCharts) return;

        // Update text metrics first
        const trainAcc = (metrics.train.accuracy * 100).toFixed(2);
        const testAcc = (metrics.test.accuracy * 100).toFixed(2);
        
        document.getElementById('train-acc').textContent = `${trainAcc}%`;
        document.getElementById('test-acc').textContent = `${testAcc}%`;
        document.getElementById('vocab-size').textContent = metrics.vocabulary_size;

        // Hide loading, show charts
        metricsContent.classList.add('hidden');
        metricsCharts.classList.remove('hidden');

        // Wait for DOM to update before creating charts
        setTimeout(() => {
            createMetricsCharts(metrics);
        }, 50);
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

function createMetricsCharts(metrics) {
    try {
        // Create line chart comparing train vs test performance
        
        // Train metrics
        const trainPrec = (metrics.train.precision * 100).toFixed(2);
        const trainRec = (metrics.train.recall * 100).toFixed(2);
        const trainF1 = (metrics.train.f1_score * 100).toFixed(2);
        
        // Test metrics
        const testPrec = (metrics.test.precision * 100).toFixed(2);
        const testRec = (metrics.test.recall * 100).toFixed(2);
        const testF1 = (metrics.test.f1_score * 100).toFixed(2);

        // Common chart options
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.8,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(241, 245, 249, 0.9)',
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        padding: 10,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Performance (%)',
                        color: 'rgba(241, 245, 249, 0.9)',
                        font: {
                            size: 12,
                            weight: '600'
                        },
                        padding: {
                            bottom: 8
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        color: 'rgba(203, 213, 225, 0.9)',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        lineWidth: 1
                    }
                },
                x: {
                    ticks: {
                        color: 'rgba(203, 213, 225, 0.9)',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        lineWidth: 1
                    }
                }
            }
        };

        // Precision Chart
        const precisionCtx = document.getElementById('precisionChart').getContext('2d');
        new Chart(precisionCtx, {
            type: 'line',
            data: {
                labels: ['Training', 'Test'],
                datasets: [{
                    label: 'Precision',
                    data: [trainPrec, testPrec],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 3,
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: commonOptions
        });

        // Recall Chart
        const recallCtx = document.getElementById('recallChart').getContext('2d');
        new Chart(recallCtx, {
            type: 'line',
            data: {
                labels: ['Training', 'Test'],
                datasets: [{
                    label: 'Recall',
                    data: [trainRec, testRec],
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 3,
                    pointBackgroundColor: 'rgba(16, 185, 129, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: commonOptions
        });

        // F1-Score Chart
        const f1Ctx = document.getElementById('f1Chart').getContext('2d');
        new Chart(f1Ctx, {
            type: 'line',
            data: {
                labels: ['Training', 'Test'],
                datasets: [{
                    label: 'F1-Score',
                    data: [trainF1, testF1],
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 3,
                    pointBackgroundColor: 'rgba(139, 92, 246, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: commonOptions
        });
    } catch (error) {
        console.error('Error creating charts:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    reviewInput.focus();
    analyzeBtn.disabled = true;
    loadMetrics();
});
