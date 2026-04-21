# Sentiment Analysis of Product Reviews on an E-commerce Website

A multi-class NLP classification system that categorizes Amazon product reviews as **positive**, **neutral**, or **negative**, comparing Multinomial Naive Bayes and Linear SVM with and without data balancing.

---

## Problem Statement

E-commerce platforms generate thousands of customer reviews daily. Manually analyzing these to understand sentiment is time-consuming and unscalable. This project builds an automated pipeline to classify reviews into three sentiment categories, helping businesses quickly identify product issues, track customer satisfaction, and make data-driven decisions.

---

## Dataset

- **Source:** [Kaggle – Amazon Review CSV](https://www.kaggle.com/datasets/busraercan/amazon-review-csv)
- **Size:** 4,915 reviews (after cleaning)
- **Features used:** `reviewText`, `summary`, `overall` (star rating)
- **Label mapping:**
  - Rating ≤ 2 → Negative
  - Rating = 3 → Neutral
  - Rating ≥ 4 → Positive
- **Class distribution:** Heavily imbalanced — positive reviews dominate

---

## Approach

1. **Data Cleaning & Preprocessing**
   - Combined `summary` and `reviewText` into a single `reviews` column
   - Lowercased, removed URLs and non-alphabetic characters
   - Removed stopwords, applied lemmatization using NLTK's `WordNetLemmatizer`

2. **Exploratory Data Analysis (EDA)**
   - Sentiment distribution (bar + pie charts)
   - Average review length by sentiment, negative reviews were longest
   - Top 20 frequent words per sentiment class
   - Word clouds for positive, neutral, and negative reviews

3. **Feature Extraction**
   - TF-IDF Vectorizer: `max_features=5000`, `ngram_range=(1,2)`, English stopwords removed
   - Fit on training set only to prevent data leakage

4. **Data Balancing**
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) on training data to balance neutral and negative classes against the majority positive class
   - Models trained both before and after SMOTE for comparison

5. **Model Training & Hyperparameter Tuning**
   - Models: Multinomial Naive Bayes and Linear SVC
   - Cross-validation: StratifiedKFold (5 folds) to preserve class ratios
   - Tuning: GridSearchCV with `scoring='f1_macro'`
     - NB: `alpha` ∈ {0.1, 0.5, 1.0}
     - SVM: `C` ∈ {0.01, 0.1, 1, 10}

6. **Evaluation**
   - Classification Report (Precision, Recall, F1-score per class)
   - Confusion matrices before and after SMOTE for both models
   - Side-by-side bar chart comparing all metrics across models

---

## Results

| Model | Accuracy | F1-Score (Macro) | Best Hyperparameter |
|-------|----------|------------------|---------------------|
| Linear SVM (after SMOTE + tuning) | 92.9% | 0.53 | C = 10 |
| Naive Bayes (after SMOTE + tuning) | 90.6% | 0.54 | α = 0.1 |

**Final model chosen: Linear SVM**, higher accuracy, better generalization, and stronger handling of high-dimensional sparse TF-IDF features.

> Both models perform strongly on the majority positive class but show lower scores on neutral due to inherent class imbalance. SMOTE improved minority class recall at the cost of a slight accuracy dip - an intentional trade-off for fairer classification.

---

## Tools & Libraries

- Python
- Scikit-learn
- NLTK
- imbalanced-learn (SMOTE)
- Pandas, NumPy
- Matplotlib, Seaborn
- WordCloud

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-ecommerce.git
   cd sentiment-analysis-ecommerce
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (first run only)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/busraercan/amazon-review-csv), place `amazon_review.csv` in the root directory

5. Run the notebook
   ```bash
   jupyter notebook Sentiment_Analysis_of_Product_Reviews_on_an_E-commerce_Website.ipynb
   ```

---

## Project Structure

```
sentiment-analysis-ecommerce/
│
├── Sentiment_Analysis_of_Product_Reviews_on_an_E-commerce_Website.ipynb
├── requirements.txt
└── README.md
```

---

## Key Takeaway

Applying SMOTE shifted both models from ignoring minority classes entirely to detecting them meaningfully — at the cost of a small accuracy drop. This illustrates a core principle in imbalanced classification: optimizing for accuracy alone is misleading when class distribution is skewed. F1-macro is the more honest metric here.
