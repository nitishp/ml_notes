# Week 1 Notes

- Machine Learning: The ability for a program to learn without being explicitly told how to

- There's a couple major types of learning algorithms
    - Supervised Learning
    - Unsupervised Learning
    - Recommender Systems
    - Reinforcement Learning

- Supervised Learning
    - Deals with X -> Y mapping: You give it a bunch of input X, and it tries to output a value Y
    - A key part is to supply this learning algorithm with a bunch of "correct" examples
    - There's two types of supervised learning:
        - Regression: Predicting continuous values (i.e. numbers)
        - Classification: Predicting which bucket a value falls into (i.e. think an enum or bool)

- Unsupervised Learning
    - You're given a set of data and just told "Hey find some structure out of this"
    - Most commonly its a clustering algorithm (group things into buckets)
    - But could also be anomaly detection or dimensionality reduction

- Linear Regression 
    - Notations
        - Training set = the "right answers" in the  data set that you train a supervised algorithm on
        - x = input / feature variable
        - y = output
        - m = # of examples in training set
        - (x <sup>(i)</sup>, y<sup>(i)</sup>) refers to the i<sup>th</sup> example
        - *f* is the model or function that your machine learning algorithm produces to predict future values
        - $\hat{y}$ is the estimated / predicted value from the function *f*
        - J(w,b) is the cost function that we compute <- want to minimize this
    - Linear regression function: *f(x)* = wx + b
        - w and b are parameters that are tweaked
    - Cost function: $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
        - You can plot J as a function of w and b and then pick the options for w and b that minimize the cost function
        - For w and b you can make a contour plot or a 3D plot to visualize the cost function

- Gradient descent
    - Algorithm to help minimize cost function
    - Intuitive explanation: At each iteration, you take a baby step in the direction with the steepest descent
    - Algorithm:
        - Do the following until convergence (i.e. values don't change):
        - Simultaneously update w and b:
            - $$tmp_w = w - \alpha * \frac{d *J(w, b)}{dw}$$
            - $$tmp_b = b - \alpha * \frac{d *J(w, b)}{db}$$
            - $$w = tmp_w$$
            - $$b = tmp_b$$
    - Reminder: Derivative of a function at a certain point is the slope at that point
    - $\alpha$ controls the learning rate
        - Too high and you might not converge
        - Too low and it might never converge
    - Gradient descent can get stuck at local minimas
        - Squared error cost function doesn't have this issue
        - Gradient descent also slows down as it approaches the optimal solution

