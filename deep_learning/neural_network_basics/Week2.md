## Logistic Regression as a Neural Network

* Used for binary classification
* Notation
    * $m$ is the number of samples
    * $X$ and $Y$ are just the individual inputs and outputs concat together column wise
        * Using a concrete example, it will show up as:
            * `X.shape = (n_x, m)` and `Y.shape = (1, m)`
    * Learn parameters $W$ and $b$
        * `W.shape = (n_x, 1)` and $b$ is a scalar
* Loss function
    $$
    Loss(\hat{y}, y) = = - (y * log(\hat{y}) + (1 - y) * log(1 - \hat{y}))
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
* Gradient descent for logistic regression - pseudo code
```
J = 0; dw_1 = 0; dw_2 = 0;
For i = 1 to m:
    z_i = w_T + b
    a_i = sigmoid(z_i)
    J += -[y_i * log(a_i) + (1 - y_i) * log(1 - a_i)]
    dz_i = a_i - y_i
    dw_1 += x_1_i * dz_i # Computed using backprop
    dw_2 += x_2_i * dz_i # Computed using backprop
    db += dz_i # Computed using backprop

J /= m
dw_1 /= m
dw_2 /= m
db /= m
```
## Vectorization 

* Really useful in deep learning because of very large datasets. Can get about a 300x speed increase
    * This is independent of using a GPU
    * Try avoid using explicit for loops in code

* Vectorized implementation of a single iteration of gradient descent
```
Z = np.matmul(w.T, X) + b
A = sigmoid(Z)
dZ = A - Y
dw = (1/m) * np.matmul(X, dZ.T)
db = (1 / m) * np.sum(dZ)
```
* Broadcasting
    * Numpy takes a number / vector and copies it along an axis
    * Example:
    ```
    [[1 2 3],
     [4 5 6]]

     +

    [100 200 300]

    = 

    [[1 2 3],
     [4 5 6]]
    
    +

    [[100 200 300],
    [100 200 300]]

    = 

    [[101 202 303],
    [104 205 306]]
    ```
* Numpy caveats
    * Example:
        * When you do a 1D vector in Numpy, the Transpose doesn't make sense
        * Example
        ```
        a = np.random.randn(5)
        print (a.shape)
        # Output (5,)

        print (a.T.shape)
        # Output (5,)

        ## NOTE This is not the same as a (5,1) or (1,5) and don't behave the same way
        ```