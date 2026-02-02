"""
Model Validation Script
=======================
This script performs comprehensive validation of your trained model:
1. Cross-Validation Check
2. Edge Case Testing
3. Inference Speed Testing

Before deployment, ensure all tests pass!
"""

import pandas as pd
import numpy as np
import time
import warnings
import joblib
warnings.filterwarnings('ignore')

# Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Model and evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

print("="*80)
print("MODEL VALIDATION SUITE")
print("="*80)
print()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'read more', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, use_lemmatization=True):
    """Complete preprocessing pipeline"""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def create_sentiment(rating):
    """Create sentiment labels from ratings"""
    if rating <= 2:
        return 0  # Negative
    elif rating >= 4:
        return 1  # Positive
    else:
        return 2  # Neutral

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

print("📁 Loading data and model...")
print("-" * 80)

# Load dataset
try:
    df = pd.read_csv('data/data.csv')
    print(f"✓ Dataset loaded: {len(df)} reviews")
except:
    print("✗ Error: Could not load data.csv")
    print("  Make sure data.csv is in the 'data' folder")
    exit(1)

# Create sentiment labels
df['sentiment'] = df['Ratings'].apply(create_sentiment)
df_binary = df[df['sentiment'] != 2].copy()

# Preprocess
print("✓ Preprocessing text...")
df_binary['cleaned_review'] = df_binary['Review text'].apply(preprocess_text)
df_binary = df_binary[df_binary['cleaned_review'].str.strip().str.len() > 0]

X = df_binary['cleaned_review']
y = df_binary['sentiment']

print(f"✓ Binary dataset size: {len(df_binary)}")
print(f"  - Positive samples: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  - Negative samples: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print()

# ============================================================================
# TEST 1: CROSS-VALIDATION CHECK
# ============================================================================

print("="*80)
print("TEST 1: CROSS-VALIDATION CHECK")
print("="*80)
print("Purpose: Ensure model performance is consistent across different data splits")
print("Target: Mean F1 > 0.85 and Std < 0.05")
print("-" * 80)

# Create vectorizer and model
print("\n🔧 Setting up model...")
bow_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_vectorized = bow_vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Perform 5-fold cross-validation
print("🔄 Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores_accuracy = cross_val_score(model, X_vectorized, y, cv=cv, scoring='accuracy', n_jobs=-1)
cv_scores_f1 = cross_val_score(model, X_vectorized, y, cv=cv, scoring='f1_weighted', n_jobs=-1)

print("\n📊 Cross-Validation Results:")
print("-" * 80)
print(f"{'Fold':<10} {'Accuracy':<15} {'F1-Score':<15}")
print("-" * 80)

for i, (acc, f1) in enumerate(zip(cv_scores_accuracy, cv_scores_f1), 1):
    print(f"Fold {i:<5} {acc:>10.4f} {f1:>15.4f}")

print("-" * 80)
print(f"{'Mean':<10} {cv_scores_accuracy.mean():>10.4f} {cv_scores_f1.mean():>15.4f}")
print(f"{'Std Dev':<10} {cv_scores_accuracy.std():>10.4f} {cv_scores_f1.std():>15.4f}")
print("-" * 80)

# Evaluation
mean_f1 = cv_scores_f1.mean()
std_f1 = cv_scores_f1.std()

print("\n✅ Cross-Validation Assessment:")
if mean_f1 > 0.90:
    print(f"   ✓ EXCELLENT: Mean F1-Score = {mean_f1:.4f} (Target: > 0.85)")
elif mean_f1 > 0.85:
    print(f"   ✓ GOOD: Mean F1-Score = {mean_f1:.4f} (Target: > 0.85)")
else:
    print(f"   ✗ NEEDS IMPROVEMENT: Mean F1-Score = {mean_f1:.4f} (Target: > 0.85)")

if std_f1 < 0.03:
    print(f"   ✓ VERY STABLE: Std Dev = {std_f1:.4f} (Target: < 0.05)")
elif std_f1 < 0.05:
    print(f"   ✓ STABLE: Std Dev = {std_f1:.4f} (Target: < 0.05)")
else:
    print(f"   ⚠ UNSTABLE: Std Dev = {std_f1:.4f} (Target: < 0.05)")
    print("   → Model performance varies too much across folds")
    print("   → Consider: More data, simpler model, or feature engineering")

print()

# Train final model for next tests
print("🔧 Training final model for edge case testing...")
model.fit(X_vectorized, y)
print("✓ Model trained successfully")
print()

# ============================================================================
# TEST 2: EDGE CASE TESTING
# ============================================================================

print("="*80)
print("TEST 2: EDGE CASE TESTING")
print("="*80)
print("Purpose: Test model behavior on unusual/extreme inputs")
print("-" * 80)

def test_edge_case(case_name, review_text, expected_behavior="Should handle gracefully"):
    """Test a single edge case"""
    print(f"\n📝 Test: {case_name}")
    print(f"   Input: '{review_text}'")
    print(f"   Expected: {expected_behavior}")
    
    try:
        # Preprocess
        start_time = time.time()
        cleaned = preprocess_text(review_text)
        
        # Handle empty cleaned text
        if not cleaned or len(cleaned.strip()) == 0:
            print(f"   ⚠ WARNING: Text became empty after preprocessing")
            print(f"   → Original: '{review_text}'")
            print(f"   → Cleaned: '{cleaned}'")
            print(f"   ✓ Handled: Would need default prediction or error message")
            return False
        
        # Vectorize
        vectorized = bow_vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        end_time = time.time()
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = max(probabilities) * 100
        
        print(f"   ✓ Prediction: {sentiment} (Confidence: {confidence:.2f}%)")
        print(f"   ✓ Processing time: {(end_time - start_time)*1000:.2f}ms")
        print(f"   ✓ Cleaned text: '{cleaned}'")
        
        return True
        
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        print(f"   → Model failed to handle this case!")
        return False

# Edge case test suite
edge_cases = [
    # Empty/minimal inputs
    ("Empty String", "", "Should detect empty input"),
    ("Whitespace Only", "   ", "Should detect empty input"),
    ("Very Short (2 words)", "Good product", "Should still predict"),
    ("Very Short (3 words)", "Very bad quality", "Should predict negative"),
    
    # Length extremes
    ("Long Review (200+ words)", 
     "This is an absolutely amazing product that exceeded all my expectations. " * 10,
     "Should handle long text efficiently"),
    
    # Special characters
    ("All Caps", "TERRIBLE PRODUCT!!! WORST PURCHASE EVER!!!", "Should handle caps"),
    ("Excessive Punctuation", "Amazing!!!! Best product!!!!!", "Should handle punctuation"),
    ("Special Characters", "Gr@#t pr0duct! W#rth $$$$", "Should clean special chars"),
    ("Emojis", "Great product 😊 👍 highly recommended! 🎉", "Should remove emojis"),
    ("Numbers Only", "12345 67890", "Should handle or reject"),
    
    # Mixed cases
    ("Mixed Language Feel", "Very good muy bueno excellent", "Should handle mixed words"),
    ("HTML Tags", "<b>Great</b> product! <a>Click here</a>", "Should remove HTML"),
    ("URLs", "Check this product at http://example.com amazing!", "Should remove URLs"),
    ("Email", "Contact support@example.com for issues", "Should remove emails"),
    
    # Real-world edge cases
    ("Only Stopwords", "the a an is was were", "Should become empty after preprocessing"),
    ("Repeated Words", "good good good good good", "Should handle repetition"),
    ("Negation", "Not good at all, terrible quality", "Should catch negative sentiment"),
    ("Sarcasm", "Oh great, another damaged product. Just wonderful.", "May struggle with sarcasm"),
]

print("\n🧪 Running Edge Case Tests...")
print("=" * 80)

passed = 0
failed = 0
warnings = 0

for case_name, review_text, expected in edge_cases:
    result = test_edge_case(case_name, review_text, expected)
    if result:
        passed += 1
    elif result is False and "empty" in case_name.lower():
        warnings += 1
    else:
        failed += 1

print("\n" + "="*80)
print("📊 Edge Case Testing Summary:")
print("-" * 80)
print(f"Total Tests: {len(edge_cases)}")
print(f"✓ Passed: {passed}")
print(f"⚠ Warnings (Empty inputs): {warnings}")
print(f"✗ Failed: {failed}")
print("-" * 80)

if failed == 0:
    print("✅ EXCELLENT: All edge cases handled successfully!")
else:
    print("⚠ ATTENTION: Some edge cases failed. Review and add error handling.")

print()

# ============================================================================
# TEST 3: INFERENCE SPEED TEST
# ============================================================================

print("="*80)
print("TEST 3: INFERENCE SPEED TEST")
print("="*80)
print("Purpose: Ensure model is fast enough for production use")
print("Targets: Single < 100ms, Batch(100) < 1 second")
print("-" * 80)

# Prepare test data
print("\n🔧 Preparing test reviews...")
test_reviews = [
    "Excellent product! Highly recommended.",
    "Poor quality. Not worth the money.",
    "Good value for price. Fast delivery.",
    "Terrible experience. Product broke on first use.",
    "Amazing quality! Will buy again.",
] * 20  # 100 reviews

print(f"✓ Test set prepared: {len(test_reviews)} reviews")

# Test 1: Single prediction speed
print("\n⏱️  Test 1: Single Prediction Speed")
print("-" * 80)

single_times = []
for i in range(10):
    review = test_reviews[i]
    
    start = time.time()
    cleaned = preprocess_text(review)
    vectorized = bow_vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)
    end = time.time()
    
    elapsed_ms = (end - start) * 1000
    single_times.append(elapsed_ms)
    print(f"   Attempt {i+1}: {elapsed_ms:.2f}ms")

avg_single = np.mean(single_times)
std_single = np.std(single_times)

print("-" * 80)
print(f"Average: {avg_single:.2f}ms ± {std_single:.2f}ms")

if avg_single < 50:
    print(f"✅ EXCELLENT: {avg_single:.2f}ms (Target: < 100ms)")
elif avg_single < 100:
    print(f"✅ GOOD: {avg_single:.2f}ms (Target: < 100ms)")
else:
    print(f"⚠ SLOW: {avg_single:.2f}ms (Target: < 100ms)")
    print("   → Consider optimizing preprocessing or using lighter model")

# Test 2: Batch prediction speed
print("\n⏱️  Test 2: Batch Prediction Speed (100 reviews)")
print("-" * 80)

batch_times = []
for attempt in range(5):
    start = time.time()
    
    # Preprocess all
    cleaned_batch = [preprocess_text(review) for review in test_reviews]
    
    # Vectorize all
    vectorized_batch = bow_vectorizer.transform(cleaned_batch)
    
    # Predict all
    predictions = model.predict(vectorized_batch)
    probabilities = model.predict_proba(vectorized_batch)
    
    end = time.time()
    elapsed_ms = (end - start) * 1000
    batch_times.append(elapsed_ms)
    
    print(f"   Attempt {attempt+1}: {elapsed_ms:.2f}ms ({elapsed_ms/len(test_reviews):.2f}ms per review)")

avg_batch = np.mean(batch_times)
avg_per_review = avg_batch / len(test_reviews)

print("-" * 80)
print(f"Average batch time: {avg_batch:.2f}ms")
print(f"Average per review: {avg_per_review:.2f}ms")

if avg_batch < 500:
    print(f"✅ EXCELLENT: {avg_batch:.2f}ms for 100 reviews (Target: < 1000ms)")
elif avg_batch < 1000:
    print(f"✅ GOOD: {avg_batch:.2f}ms for 100 reviews (Target: < 1000ms)")
else:
    print(f"⚠ SLOW: {avg_batch:.2f}ms for 100 reviews (Target: < 1000ms)")

# Test 3: Preprocessing bottleneck analysis
print("\n⏱️  Test 3: Bottleneck Analysis")
print("-" * 80)

sample_review = "Excellent product! Very good quality and fast delivery. Highly recommended!"

# Time preprocessing
start = time.time()
cleaned = preprocess_text(sample_review)
prep_time = (time.time() - start) * 1000

# Time vectorization
start = time.time()
vectorized = bow_vectorizer.transform([cleaned])
vec_time = (time.time() - start) * 1000

# Time prediction
start = time.time()
prediction = model.predict(vectorized)
probabilities = model.predict_proba(vectorized)
pred_time = (time.time() - start) * 1000

total_time = prep_time + vec_time + pred_time

print(f"Preprocessing: {prep_time:.2f}ms")
print(f"Vectorization: {vec_time:.2f}ms")
print(f"Prediction:    {pred_time:.2f}ms")
print(f"Total:         {total_time:.2f}ms")

if total_time > 0:
    print(f"Preprocessing share: {(prep_time/total_time)*100:.1f}%")
    print(f"Vectorization share: {(vec_time/total_time)*100:.1f}%")
    print(f"Prediction share:    {(pred_time/total_time)*100:.1f}%")
else:
    print("⚠ Timing too fast to compute percentage breakdown reliably.")


print("\n💡 Optimization Tips:")

if total_time > 0:
    if prep_time / total_time > 0.5:
        print("   → Preprocessing is the bottleneck (>50% of time)")
        print("   → Consider: Caching common operations, optimize regex")
    elif vec_time / total_time > 0.3:
        print("   → Vectorization takes significant time")
        print("   → Consider: Reduce max_features or use simpler n-grams")
    else:
        print("   ✓ Well balanced pipeline - no major bottlenecks")
else:
    print("   ⚠ Timing resolution too low to identify bottlenecks.")
    print("   ✓ Pipeline is extremely fast; no optimization required.")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("VALIDATION SUMMARY")
print("="*80)

print("\n📊 Overall Results:")
print("-" * 80)

# Cross-validation
cv_status = "✅ PASS" if mean_f1 > 0.85 and std_f1 < 0.05 else "⚠ REVIEW"
print(f"1. Cross-Validation:     {cv_status}")
print(f"   - Mean F1-Score: {mean_f1:.4f}")
print(f"   - Std Deviation: {std_f1:.4f}")

# Edge cases
edge_status = "✅ PASS" if failed == 0 else "⚠ REVIEW"
print(f"\n2. Edge Case Testing:    {edge_status}")
print(f"   - Passed: {passed}/{len(edge_cases)}")
print(f"   - Failed: {failed}/{len(edge_cases)}")

# Speed
speed_status = "✅ PASS" if avg_single < 100 and avg_batch < 1000 else "⚠ REVIEW"
print(f"\n3. Inference Speed:      {speed_status}")
print(f"   - Single prediction: {avg_single:.2f}ms (Target: <100ms)")
print(f"   - Batch (100): {avg_batch:.2f}ms (Target: <1000ms)")

print("-" * 80)

# Overall assessment
all_pass = (mean_f1 > 0.85 and std_f1 < 0.05 and 
            failed == 0 and 
            avg_single < 100 and avg_batch < 1000)

if all_pass:
    print("\n🎉 READY FOR DEPLOYMENT!")
    print("   All validation tests passed successfully.")
    print("   Your model is stable, handles edge cases, and is fast enough.")
elif mean_f1 > 0.85 and avg_single < 100:
    print("\n✅ READY FOR DEPLOYMENT (with minor notes)")
    print("   Model performance is good. Address warnings before production.")
else:
    print("\n⚠ NEEDS IMPROVEMENT")
    print("   Review failed tests and optimize before deployment.")

print("\n" + "="*80)
print("Validation complete! Review the results above.")
print("="*80)
