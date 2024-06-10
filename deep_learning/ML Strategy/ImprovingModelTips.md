## Introduction
* If you want to improve an ML model, there's a lot of things you can change:
  * The number of layers, number of hidden units, optimization algorithm, collecting more data etc.
  * It's helpful to have a framework for deciding what to do next
* Orthogonalization
  * You want to divide which methods to try for which problem:
    * The method used to tune a model to perform better on the training set is very different from performing well on the dev or test test

## Setting up your goal
* Speed up iteration of the ML loop by having a single real number evaluation metric
  * Something like an F1 score is easier to compare different classifiers to compare performance on the dev set
  * If you have $N$ metrics:
    * It might make sense to have 1 optimizing metric. This is one you want to do as well as possible.
    * And $N - 1$ satificing metrics. This is one where you can just do good enough

## Train/dev/test distributions
* Have your test and dev sets come from the same distribution. Randomly shuffle data to distribute it evenly
* Split ratios:
  * If you have something like 10,000 examples, splitting into 60/20/20 as a train/dev/test split
  * If you have something like 10M examples, splitting into 98/1/1 as a train/dev/test split
* When to change your evaluation metrics or test set:
  * If the evaluation metric in the real world changes
  * If the distribution of data changes
  * When doing this, separate out defining the evaluation metric vs actually doing well on the metrics

## Comparing to human-level performance
* Bayes error: The lowest possible error that an ML algorithm can achieve
  * This is better than human level performance
* Comparing to human level performance is good because:
  * Humans can help you label (x, y) dataset
  * Can do manual error analysis: Why did a human get this right? 
  * Tells you to focus on bias or variance
    * Human level performance can often be a proxy for bayes error
    * If the training set error rate is vastly different from the human level error rate ("Avoidable bias"), you know there's a lot of bias in the algorithm
    * If the dev set error rate is higher than the training set error rate, it could be due to variance
* Remember that your model's error cannot be lower than Bayes error, if it is then, either:
  * The model is overfitting to the training data
  * The bayes error is actually lower than you think
