# Week 2 Notes

- Multiple Linear Regression
    - $$f_w,b(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b = w \cdot x + b$$
    - x<sup>(2)</sup><sub>4</sub> refers to the 4th feature in the 2nd example (which is normally the row)
    - Vectorized code: `f = np.dot(w, x) + b`
        - Runs faster because it can break this up into parallel processes

- Gradient descent still works the same, the only difference is in the derivative term

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

- Feature scaling
    - General Tip: Try to keep feature values in similar ranges (normally roughly between -1 and 1)
    - Having imbalanced feature ranges can cause gradient descent to run more slowly
        - Intuitively if a feature has a large range, it'll need a small weight value (or if a feature has a small range, it'll need a large weight value)
            - From a mathematical sense, in the gradient descent step, the update step takes the value of the derivative and multiplies it by the feature. If the feature value is really large, the update step updates a large number
            - Gradient descent will bounce around its optimal value, you'll see the derivative term flipping signs
        - Then on each step of gradient descent an unoptimal value is picked for the smaller weight value
        - You have a couple options:
            - Mean normalization: Subtract from the mean and divide by the range
            - Divide by the largest value
    - You have to normalize the values of any new inputs that you want to predict

- Gradient descent cost should decrease on every iteration of the algorithm
    - If it goes up the learning rate is too large
- Try running gradient descent for a small number of iterations first, find out the highest rate for which the cost keeps going down and then pick that as the learning rate

- Feature Engineering
    - The act of combining / transforming existing features to change the model (i.e. the new model can look like):
    $$f(x) = w_1 *x_1 + w_2*x_2 + w_3*(x_1 * x_2) + b$$
    - This can also be used for polynomial regression like so:
    $$f(x) = w_1 * x_1 + w_2*(x_1)^2 + b$$
    - When feature engineering, feature scaling becomes even more important
