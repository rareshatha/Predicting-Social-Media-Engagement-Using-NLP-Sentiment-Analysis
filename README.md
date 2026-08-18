# Predicting Social Media Engagement Using NLP & Sentiment Analysis

## Overview

An NLP-based machine learning project that predicts social media engagement levels from text while analyzing the underlying sentiment of posts.

The project compares traditional machine learning approaches using TF-IDF with a fine-tuned BERT model, and introduces a hybrid sentiment-enhanced classification approach to improve predictions for borderline cases.

## Objectives

* Predict high vs. low social media engagement from text.
* Analyze sentiment and linguistic patterns in social media content.
* Compare traditional machine learning models with transformer-based deep learning.
* Evaluate model performance using Accuracy, Precision, Recall, and F1-Score.
* Explore ethical considerations related to privacy, bias, and fairness in automated social media analysis.

## Dataset

The project uses the `Shekswess/social-media-instruction` dataset from Hugging Face. The dataset contains message-based text interactions stored in nested JSON-like structures.

The text data was flattened and transformed into usable features for NLP analysis.

## NLP Pipeline

1. Exploratory Data Analysis (EDA)
2. Data cleaning and preprocessing
3. JSON/text flattening
4. Duplicate and missing-value handling
5. Sentiment analysis
6. TF-IDF feature extraction
7. Engagement score generation
8. Model training and hyperparameter tuning
9. BERT fine-tuning
10. Hybrid sentiment-enhanced classification
11. Model evaluation and comparison

## Models

### Traditional Machine Learning

* TF-IDF + Random Forest
* TF-IDF + Logistic Regression

### Deep Learning

* Fine-tuned `bert-base-uncased`

### Hybrid Approach

A VADER sentiment analysis layer was integrated with the BERT classifier. When BERT confidence was below 0.7, sentiment-based rules were used as a fallback for classification.

## Results

| Model               |   Accuracy | Precision |     Recall |   F1-Score |
| ------------------- | ---------: | --------: | ---------: | ---------: |
| Random Forest       |     74.67% |    70.58% |     72.72% |     71.64% |
| Logistic Regression |     74.67% |    70.58% |     72.72% |     71.64% |
| BERT                | **76.00%** |    70.27% | **78.78%** | **74.28%** |

BERT achieved the best overall performance, particularly in recall, demonstrating the benefit of contextual language modeling for social media engagement prediction.

## Technologies

* Python
* Natural Language Processing (NLP)
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* NLTK / VADER
* TextBlob
* TF-IDF
* BERT
* Pandas
* Regular Expressions

## Ethical Considerations

The project considered privacy, bias, and fairness during the analysis. User IDs and metadata were removed during preprocessing, and the analysis focused on publicly available message content. Sentiment ranges were also evaluated to reduce the risk of biased predictions.

## Future Improvements

* Train BERT for longer using stronger GPU resources.
* Incorporate temporal engagement features such as posting time and response latency.
* Introduce more granular engagement labels.
* Explore multimodal models combining text with images and video.
* Deploy the model as a real-time API for content analysis and A/B testing.

## Academic Context

Developed as part of **AI312 – Natural Language Processing** at the University of Prince Mugrin.

**Project Title:** Predicting Social Media Engagement Using NLP and Sentiment Analysis
