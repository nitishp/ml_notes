# Week 2 Notes

- Multiple Linear Regression
    - $$f_w,b(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b = w \cdot x + b$$
    - x<sup>(2)</sup><sub>4</sub> refers to the 4th feature in the 2nd example (which is normally the row)
    - Vectorized code: `f = np.dot(w, x) + b`
        - Runs faster because it can break this up into parallel processes

- Gradient descent still works the same, the only difference is in the derivative term

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$
