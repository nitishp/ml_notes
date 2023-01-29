# Week 3

- We'll learn about a set of diagnostics we can use to determine what to do next to improve our algorithm's effectiveness

- Evaluating a model
    - Split your data up into a test set and a training set (maybe a 70/30 or 80/20 split)
    - Train the model over the training set
    - You can then compute the cost over the training set called ($J_{test}$)
    - You can evaluate the cost $J_{test}$ over the test set to estimate how accurate the model you think will be in real life
        - Note that when evaluating, don't use the regularization term
    - For classification problems, it's also more common to just look at the percentage of things misclassified as a way of gauging performance
- Picking between multiple models
    - One approach you could do is to make multiple models, compute $J_{test}$ for each, and then pick the model with the lowest $J_{test}$ value.
        - This is flawed
        - You're using $J_{test}$ to make a decision about your machine learning model, and so your $J_{test}$ won't be as accurate as in real life
    - What you should do instead is to split up your data into a training set, cross validation set and test set
    - You use the training set as normal
    - You use the cross validation set to pick the best model
    - You use the test set for reporting

