## Logistic Regression as a Neural Network

* Used for binary classification
* Notation
    * $X$ and $Y$ are just the individual inputs and outputs concat together column wise
        * Using a concrete example, it will show up as:
            * `X.shape = (n_x, m)` and `Y.shape = (1, m)`
    * Learn parameters $W$ and $b$
        * `W.shape = (n_x, 1)` and $b$ is a scalar
* Loss function
    $$
    Loss(\hat{y}, y) = = - (y * log(\hat(y)) + (1 - y) * log(1 - \hat{y})
    $$
* Cost function
    $$
    J(w, b) = \frac{1}{m}\sum_{i = 1}^{m} Loss(\hat{y}, y)
    $$
* Gradient descent recap
    $$
        w = w - \alpha\frac{\partial{J(w)}}{\partial{w}}
    $$
    * $\alpha$ is the learning rate
* Computation Graph and backward propogation calculation
    * Check out /course_2/Week_2_Notes.md
