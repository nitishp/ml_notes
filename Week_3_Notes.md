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
