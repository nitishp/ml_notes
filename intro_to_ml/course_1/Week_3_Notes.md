# Week 3 Notes
Classification
- Output values into a "class" or output category instead of predicting a continuous range of values
- Why won't linear regression work here?
  - You could in theory fit a best fit line and pick an arbitrary threshold, say 0.5, and say any predicted value below this gets the value of 0
  - This won't work if there's a strong outlier (since that changes the best fit line, even when it shouldn't change the prediction threshold)
  - It's easier to see with an example presented in lecture with the training slides (predicting tumor size vs if its malignant or not). If the threshold was say tumor size < .3  results in a negative prediction reasonably well, having a tumor of size 1000 throws off the best fit line to predict things incorrectly (it moves the threshold way to the right, but it shouldn't, since its a "good" outlier in this case)

- Logistic Regression
  - Core function:
    $$ z = w \cdot x + b $$
    $$ g(z) = \frac {1} {1 + e^{-z}} $$

    - The second function is called the sigmoid function
    - It will give you the probability if the classification is 0 or 1
      - You still pick some threshold value and choose based on that

  - Decision Boundary
    - You can choose a threshold value above which $\hat{y}$ is always classified as 1
      - It's up to you what threshold to pick, but a common one is 0.5
      - The decision boundary then becomes the when $ g(z) = threshold value$
    - If you pick a threshold of say 0.5, it'll be when $z \geq 0$, which happens to be when $ w \cdot x + b \geq 0 $
    - Note that $z$ doesn't need to be a linear function, it could be a complex nonlinear function like this
      $$ z = w_1x_1 + w_2x_2x_1 + w_3x_2^2 + b $$
    - This can help create complex non-linear functions like circles or ellipses

- Cost function
  - The mean squared error cost function won't work because it's not a convex function for our logistic regression f(x)
  - Instead let's use a different cost function like this:
    - $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
    - It looks complicated, but recall that $y^{(i)}$ can only be 0 or 1
      - In the case that $y^{(i)} = 1$:
        - If the prediction $f_{w,b}(x^{(i)}) \rightarrow 1$, the $Loss() \rightarrow 0$ which makes sense since the prediction is accurate
        - If the prediction $f_{w,b}(x^{(i)}) \rightarrow 0$, the $Loss() \rightarrow \infty$ which makes sense since the prediction is completely wrong
      - In the case that $y^{(i)} = 0$:
        - If the prediction $f_{w,b}(x^{(i)}) \rightarrow 0$, the $Loss() \rightarrow 0$ which makes sense since the prediction is accurate
        - If the prediction $f_{w,b}(x^{(i)}) \rightarrow 1$, the $Loss() \rightarrow \infty$ which makes sense since the prediction is completely wrong
    - Recall that the cost function is just the sum of all of the Loss functions
      - $$J_{w,b}(f(x)) = \frac{1}{m}\sum_{j=1}^{m} Loss(f_{w,b}(x^{(i)}, y^{(i)})) $$

- Gradient descent
  - Same concepts still apply actually, the derivatives even **look** the same:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$


$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
\end{align}
$$

  - Note though that it is different because the the $f_{w,b}(x^{(i)})$ is different and it's a sigmoid function instead
  - All the usual stuff about feature scaling still applies!

- Overfitting / Underfitting
  - Underfitting / high bias
    - Occurs when the predicted model does not fit the training set very well
    - Also called "High bias" because the model has a highly biased / preconcieved notion of what the data looks like (e.g. always assuming its a straight line)
  - Overfitting / high variance
    - Occurs when the predicted model fits the training set really well but fails to generalize well to other examples
    - Called high variance because the predicted model can vary greatly depending on the training set (i.e. if one example was different in the training set then the entire predicted model would be different)


- Ways to fix overfitting
  - Collect more training data - This will make the model have more data to make correct predictions
  - Feature selection - Reduce the number of features that you take in into your predicted algorithm (i.e. make it a bit simpler)
  - Regularization - Introduce a bit of an error term to make it so that the weight values aren't entirely set to 0 (eliminated like in feature selection), but reduced to reduce their impact

- Regularization
  - Intuition: If we modify the cost function to add in terms to increase the cost for high values of $w_j$, then in theory the algorithm would be disincentivized from picking high values of $w_j$.
  - The cost function for linear regression and logistic regression with regularization still remains the same, but with a regularization term added:
  $$J_{w,b}(f(x)) = \frac{1}{m}\sum_{i=1}^{m} Loss(f_{w,b}(x^{(i)}, y^{(i)})) + \frac{\lambda}{2m}\sum_{j=1}^{n}(w_j)^2$$
  - $\lambda$ is called the regularization term. It controls a tradeoff between two factors in the cost function (how well we want to fit the training data and how smooth we want the function to be)
  - We don't regularize the term $b$ (you can but it doesn't make too big a difference in practice)
  - The partial derivative terms look the same except for $\frac{\partial J(\mathbf{w},b)}{\partial b}$ which now looks like:
    $$
      \frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} + \frac{\lambda}{m}w_j
    $$
  - Note these look the same for linear and logistic regression, but the main difference between these two types of algorithms is that the $f_{w,b}$ functions are completely different (making the equations different)
