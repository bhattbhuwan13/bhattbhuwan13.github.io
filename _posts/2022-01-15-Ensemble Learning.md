---
layout: post
title: Ensemble Learning
tags: ensemble machine-learning decision-tree random-forest
---

# Introduction

[Bias variance tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff) is one fo the well known problem in machine learning. According to wikipedia,  

   >**Bias–Variance problem** is the conflict in trying to simultaneously minimize these two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.
    
Two ways to tackle the bias variance tradeoff are:  
1. Change methodoligies used to create models. i.e depending upon the issue(high bias or high variance) use appropriate methods like using complex model, adding features, add training data, reduce features, etc.   
2. Use ensemble learning

In this blog, we will focus on the 2nd method.  


# Ensemble learning
Ensemble learning is the method of systematically reducing the bias and variance by combining the output of multiple models either by majority voting or by taking weighted sum. However, we cannot combine any model to form an ensemble. To form an ensemble whose collective accuracy is better than using an individual model, the individual models in an ensemble should be,    

- Accurate: The accuracy of all models should be better than random guessing. For binary classification, the accuracy of each model should be greater than 50%.   
- Diverse: The models should be diverse. Models participating in an ensemble shouldn't make errors on the same subset of data. Otherwise, the ensemble will be no better than an individual model.   

Some of the commonly used techniques to create an ensemble are described below.   

## Bagging

Bagging, or bootstrap aggregating, combines models with low bias(strong learners) and reduces their variance. From a single dataset $D$, it employs bootstrapping or random sampling with replacement to derive multiple datasets: $D_1, D_2, ... D_m$ to train $m$ different models. The output of these $m$, models is then combined by either majority voting(for classification) or by averaging(for regression) to obtain the final output of the ensemble.  

## Boosting
Boosting is the technique of combining multiple weak(high bias) models in a sequence such that each of the model is trained to reduce the error made by the previous model of the sequence. Since models are trained to reduce the errors made by previous models in the sequence, there is no need for bootstrapping and hence a single dataset is sufficient. Adaptive boosting and gradient boosting are some of the most commonly used boosting techniques.  

# Hands-on Bagging
Now, we will use scikit-learn to create a bagging classifier to classify iris-species. Let's get started.  

### Importing required libraries and Dataset preparation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
```
Let's load the iris dataset from the `datasets` module in `scikit-learn`
.
```python
iris = datasets.load_iris()
X = iris.data
y = iris.target
```
We now split the data into train and test set. The model will later be trained on the train set and evaluated on the test set.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Training the decision tree classifier
First, we will train a decision tree and later compare it with an ensemble formed by bagging of decision tree models. 
```python
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train),classifier.score(X_test, y_test)
```

```bash
Output:  
(1.0, 1.0)
```
### Training an ensemble of decision trees
Now, we will train a bagged model which will contain a collection of decision trees. 
```python
ensemble = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100,
                            bootstrap=True, random_state=42)
ensemble.fit(X_train, y_train)
ensemble.score(X_train,y_train),bag_clf.score(X_test,y_test)
```

## References

1. [Going Deeper with Convolutions, Szegedy et. al.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)

2. [Deep Residual Learning for Image Recognition, He et. al.](https://arxiv.org/pdf/1512.03385.pdf)
