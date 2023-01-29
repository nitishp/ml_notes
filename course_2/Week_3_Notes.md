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

- When using feature scaling, be sure to use the mean and standard deviation of the **training** set. This is because these are the mean and training set that the model was trained on

- Bias and Variance
    - High Bias - Underfitting
        - $J_{train}$ and $J_{cv}$ are both high
    - High variance - Overfitting
        - $J_{train} << J_{cv}$
    - You can pick a good parameter or good model based on these two things
        - Try out a bunch of models and $\lambda$ values, and pick the one with the lowest $J_{cv}$
    - How do you know if $J_{train}$ or $J_{cv}$ is high. 
        - It's good to establish a baseline level of performance that you hope to achieve
            - You can use human performance as a benchmark for this
            - You can also use competitor performance for this
        - If $J_{train}$ is much higher than the baseline, then you know you have high bias
        - If $J_{train}$ is close to the baseline but $J_{cv}$ is high, you know you have high variance
    - Learning curves
        - Plot $J_{train}$ and $J_{cv}$ as a function of the number of training examples
        - For high bias, adding more training examples (alone) won't help
        - For high variance, adding more training examples will help since it helps the overfit curve work better with the additional data
    - Neural networks
        - Simple recipe
            - Train a neural network with data
                - Measure $J_{train}$ to baseline:
                    - If too high, make neural network bigger and retrain
                    - If low, move on to the next step
                - Compare $J_{cv}$ to $J_{train}$
                    - If $J_{cv}$ is much higher, you have high variance
                        - Get more data and go back to step one
                    - Otherwise you're done
        - This has two caveats:
            - Training more complicated models takes more time
            - Getting more data may not always be easy
        - Neural networks that are large don't suffer from overfitting so long as they're regularized properly
        - It can be done in tensorflow like so:
            ```
            model = Sequential([
                Dense(units=25, activation='relu', kernel_regularization=L2(0.01)),
                Dense(units=12, activation='relu', kernel_regularization=L2(0.01))
                Dense(units=1, activation='linear', kernel_regularization=L2(0.01))
            ])
            ```
            - The 0.01 here is the $\lambda$
