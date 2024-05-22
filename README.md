# Advanced Email Filtering with Machine Learning

## Project Overview

The goal of this project is to create visualizations, explore text transformations or data processing techniques, and select the best machine learning model and the best hyperparameters to obtain the highest possible accuracy and recall for spam email detection.

## Dataset Description

**Spam Filter: Identifying Spam Using Emails**

**Context**: The dataset includes information that helps identify spam emails. It was acquired from 'Karthickveerakumar' and represents emails collected over a specific time period, containing key attributes necessary for spam detection.

**Owner of the dataset**: [Karthickveerakumar](https://www.kaggle.com/karthickveerakumar)

The primary objective is to ensure that the model effectively identifies spam emails, focusing on Recall as the primary metric to minimize the number of actual spam emails classified as not spam (false negatives). A high recall score indicates that the model is able to identify most of the actual spam emails (Positive), keeping the number of false negatives low.

## Text Processing

**Stop Words**: Commonly used words in a language that are often filtered out or ignored in natural language processing (NLP) tasks (e.g., "the," "is," "in," "and," "to," "a"). Removing stop words helps reduce dimensionality and makes text processing tasks more efficient by focusing on words that carry more meaningful information.

**TF-IDF Vectorization**: Techniques like tokenization, stemming, lemmatization, and vectorization (e.g., TF-IDF) are used to preprocess the text data. The TF-IDF vectorizer converts the cleaned text data into numerical features, ignoring terms that appear in more than 85% of the documents or fewer than 5 documents, and keeping only the top 10000 features. The English stop words list is used to remove common words that are not useful for analysis.

## Model Building

**Model Selection**: Choosing the appropriate machine learning models for spam detection is critical to the success of this project. The models considered include:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

The primary objective is to ensure that the model effectively identifies spam emails, focusing on Recall as the primary metric to minimize false negatives.

**Training the Models**: The data was split into training and test sets and various models were trained using cross-validation.

## Results

**Best Model Based on Recall**: The Support Vector Machine (SVM) emerged as the best model based on the Recall metric, with a Recall of 0.9748. This indicates that the SVM model is highly effective at identifying spam emails, minimizing the number of false negatives. Recall is particularly important in spam detection as it ensures that most spam emails are correctly identified.

Following the SVM, the Random Forest classifier performed well with a Recall of 0.9608, making it the second-best model in terms of recall.

### Explanation of Metrics

- **Accuracy**: Measures the overall correctness of the model. It is the ratio of correctly predicted instances (both spam and not spam) to the total instances.
- **Precision**: Indicates the accuracy of the positive predictions (spam). It is the ratio of true positive predictions to the total positive predictions (both true and false positives).
- **Recall**: Measures the ability of the model to identify all relevant instances (spam). It is the ratio of true positive predictions to the total actual positives (both true positives and false negatives).
- **F1 Score**: Harmonic mean of precision and recall. It balances the two metrics, providing a single measure of a model's performance.

In summary, while the Support Vector Machine offers the highest recall, ensuring most spam emails are correctly flagged, it also exhibits high values in other metrics, making it a robust choice. The Random Forest classifier also provides strong performance, particularly in recall and precision, making it a reliable alternative.

## Model Optimization

**Hyperparameter Tuning**: To optimize the model parameters, Grid Search was utilized for both the Support Vector Machine (SVM) and Random Forest classifiers. The goal was to find the best parameters (based on the 'recall' scoring) that maximize the model's ability to identify spam emails.

**Cross-Validation**: Cross-validation was implemented to ensure the robustness and generalization of the models. By evaluating multiple metrics (accuracy, precision, recall, F1 score) during the hyperparameter tuning process, the models were ensured to not only perform well in terms of recall but also maintain a balance with precision.
