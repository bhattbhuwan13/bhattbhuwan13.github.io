---
layout: post
title: Ensemble Learning
tags: ensemble machine-learning decision-tree random-forest
---


[Bias variance tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff) is one of the well known problem in machine learning. According to wikipedia,  

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

*Note: Stacking is also a method to form an ensemble but, we won't discuss that here.*

# Hands-on Bagging
Now, we will use scikit-learn to create a bagging classifier to classify iris-species. Let's get started.  

### Importing required libraries and Dataset preparation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets

from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

### Training the decision tree classifier
First, we will train a decision tree and later compare it with an ensemble formed by bagging of decision tree models. 
```python
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

print("Train accuracy of decision tree = ", decision_tree_classifier.score(X_train, y_train))
print("Test accuracy of decision tree = ", decision_tree_classifier.score(X_test, y_test))
```
**Output**  
```bash
Train accuracy of decision tree =  1.0
Test accuracy of decision tree =  0.95
```
We can see that the decision tree model clearly overfits. The train accuracy is 100% while the accuracy on the test set is only 95%.  Next, we will train and evaluate an ensemble.  

### Training an ensemble of decision trees
Now, we will train a bagged model which will contain 3 different decision trees.  
```python
ensemble = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=3,
                            bootstrap=True, random_state=42)
ensemble.fit(X_train, y_train)

print("Train accuracy of ensemble = ", ensemble.score(X_train, y_train))
print("Test accuracy of ensemble = ", ensemble.score(X_test, y_test))
```

**Output**  
```bash
Train accuracy of decision tree =  0.98
Test accuracy of decision tree =  0.95
```

We can see that, the even though the bagged model isn't 100% accurate in the train set, its accuracy is still 95% on the test set. The ensemble model has reduced the overfitting. The iris-dataset contains only 150 instances of data. If we had a larger dataset, the positive effects of training an ensemble model would be more apparent.  
## References

1. [Ensemble Methods in Machine Learning, Thomas G. Dietterich](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf)

2. [A Study Of Ensemble Methods In Machine Learnin, Kim et. al.](http://cs229.stanford.edu/proj2015/222_report.pdf)
