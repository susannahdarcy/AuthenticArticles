#!/usr/bin/env python
# coding: utf-8

'''
---------------------
Susannah D'Arcy Take Home Assignment - Authentic Classifier

To identify articles which are authentic (or not) I will be using an SVM Classifier (SGDClassifier).
As SVM's have shown to be able to generalise well in high-dimensional feature spaces. Which is crucial for textual classification.
I will feed the model a Tf-IDF Vectorized training set, identifying repeated words in the dataset.
---------------------
'''
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt 

raw_trainDF = pd.read_csv('train.csv')
raw_testDF = pd.read_csv('test.csv')
raw_labelsDF = pd.read_csv('labels.csv')

# Combine test and labels to get the test set.
raw_testSetDF = pd.merge(raw_testDF, raw_labelsDF, on='id')

'''
---------------------
Preprocessing

- Remove rows with null 'text'
- Replace null 'titles' and 'author' with the string 'none'
- Combine each textual column into one text variable
- Vectorise the text variable using a Tf-IDF vectorizer

I choose to keep titles and authors (and replace null values with none)
The language used in the title and possible repeated authors could be a key identifier for the classifier.
Such as more 'clickbaity' language being used in unreliable articles.
---------------------
'''
print("Before Cleaning")
print(raw_trainDF.isnull().sum())
print(raw_testSetDF.isnull().sum())

trainDF = raw_trainDF[raw_trainDF['text'].notnull()]
testDF = raw_testSetDF[raw_testSetDF['text'].notnull()]

trainDF = trainDF.fillna('none')
testDF = testDF.fillna('none')

print("\nAfter Cleaning")
print(trainDF.isnull().sum())
print(testDF.isnull().sum())


# Check if the training set is balanced.
print(trainDF['label'].value_counts().plot.pie(autopct='%1.1f%%'))
# The pie chart shows an evenly balanced dataset, which reduces the chance of a biased classifier.
# If it was unbalanced, I would have used SMOTE upsampling to reduce bias.


# TfidfVectorizer only takes 1 string, therefore I  will combine each into a textual column
combined_trainDF = pd.DataFrame(trainDF['label'])
combined_trainDF['text'] = trainDF['title'] + ' - ' + trainDF['author'] + ' : ' + trainDF['text']

combined_testDF = pd.DataFrame(testDF['label'])
combined_testDF['text'] = testDF['title'] + ' - ' + testDF['author'] + ' : ' + testDF['text']


'''
---------------------
Training 
To get an optimal classifier, I will be conducting a grid search on the input parameters for both TfidfVectorizer and SGDClassifier
We will search different max_df values, which is the cut-off value for terms with high frequency
Also, alter different regularization for the SGD (alpha and penalty).

To conduct a grid search on both the pre-processing/transformation step and the classifier, I will use a pipeline.
---------------------
'''

# Transfer training sets into numpy arrays 
y_train = combined_trainDF.pop('label').values
y_class = ['reliable', 'unreliable']

X_train = combined_trainDF.values.flatten()

y_test = testDF.pop('label').values
X_test = testDF['text'].values.flatten()

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

pipeline = Pipeline(
    [
        ("vtfidf", TfidfVectorizer()),
        ("clf", SGDClassifier()),
    ]
)

parameters = {
    "vtfidf__stop_words": ["english"],
    "vtfidf__max_df": (0.7, 0.8, 0.9, 1.0),
    "clf__alpha": (0.00001, 0.000001),
    "clf__penalty": ("l2", "elasticnet")
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=1,
                           cv=10, scoring=scoring, n_jobs=-1, refit="Accuracy")
best_model = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print(best_parameters)


'''
---------------------
Evaluation
I will now use the best pipeline model to vectorise the test set and then make a prediction.
If we were to fit a new TfidfVectorizer for the test set, due to the test set being smaller it could result in a poorer performance
IF I combined the two sets (training and testing) then train the TfidfVectorizer. It could result in a data leak,
and reduce the model prediction for unseen data.

To evaluate the prediction I will be using accuracy, F1-Score and a confusion matrix.
---------------------
'''

# Vectorise the test set, by skipping the final estimator in the pipeline.
vtfidf_X_test = best_model.best_estimator_[:-1].transform(X_test)
# best_estimator_.predict recalls the transformation step. Therefore we skip this and only call the last element of the pipe
y_pred = best_model.best_estimator_[-1].predict(vtfidf_X_test)

acc=accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3%}")

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3}")

'''
We have an accuracy of 64%, which is a fairly poor performance. F1 Score is 0.66, therefore an above-average precision and recall
This shows that this model isn't enough to consistently predict authentic articles. 
To validate this theory we could add different classifiers into the grid search to find the best model.
However this performance could be due to the test/train split
Therefore if I had access to the whole dataset I could introduce cross-validation to determine the best split.
'''

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

'''
From the confusion matrix we can see that we have a fairly even split amongst the correct types (tp, tn) and error types(fp,fn)
We have slightly more false positives which can suggest that our model is too aggressive, 
and therefore increasing the penalty could improve the model 
'''

'''
---------------------
Improvements
As stated before, adding different models to the pipeline and grid search could find a more accurate classifier for this dataset

Another method is to use Google's Deep learning NLP Bert. With this, I could have tokenised the data with Bert's Tokenizer
Then wrap the tokenized text into a torch dataset. Which PyTorch could use to train the BertForSequenceClassification.
This most likely would have resulted in a better performing model, however, as the cost of increased computation power is needed.
Therefore due to time constraints, I decided to use a more classic sklearn model.
---------------------
'''





