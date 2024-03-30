* Applied ML is a highly iterative process
  * Number of layers, number of hidden units, learning rates are all things you have to really experiment with
  * Intuitions from one domain (NLP, Computer Vision) don't translate to other domains
  * No one really picks the best hyperparameters the first time
* Ratios for splitting into train/dev/test sets are very different based on how much data you have
  * Definitions
    * Training set: Used to train model
    * Dev set: Used to evaluate different models
    * Test set: Used for final reporting of error rates
  * If you have a lot of data (say millions of examples), its okay to have dev and test sets be a lot smaller (say 1% of the entire dataset)
  * In general, try to keep the dev/test sets as close as possible to production data to have a better sense of model accuracy in the real world

* High bias = underfitting the data
  * It's "Biased" a lot by the model
  * High training set error, high dev set error
  * Fixes: Bigger network, different neural network architecture
* High variance = overfitting the data
  * It "varies" a lot with the training data
  * Low training set error, high dev set error
  * Fixes: More data, regularization
* It's possible to have both high bias and high variance. It means your model is bad and it's overfitting the training data

# Regularization
* Regular regularization equation for logistic regression
$$
J(w, b) = \frac{1}{m} \sum_{i = 1}^{m}Loss(\hat{y}, y) + \frac{\lambda}{2m}||w||_{2}
$$
Where:
$$
||w||_{2} = \sum_{i=1}^{n_x}w_{i}^2
$$
* Regularization with neural networks
$$
J(w, b) = \frac{1}{m} \sum_{i = 1}^{m}Loss(\hat{y}, y) + \frac{\lambda}{2m}\sum_{l = 1}^{L}||w^l||_{2}
$$
* Where the regularization term is the Frobenius matrix:
$$
||w^l||_2 = \sum_{i=1}^{n_l}\sum_{j=1}^{n_{l-1}}w_{i,j}^2
$$
* On backprop:
$$
dW^l = (same\,as\,backprop) + \frac{\lambda}{m}W^l
$$
$$
W^l = W^l - \alpha * dW^l
$$
$$
W^l = W^l - \frac{\alpha\lambda}{m}W^l - (same\,as\,backprop)
$$

Notice that this leads to the same equation as before but you're subtracting the same value from a smaller value on each iteration. This is why it's called "weight decay"

* Intuition
  * If we want to minimize the cost function, and we set $\lambda$ to be really high, we'll see that it incentivizes $W$ to be really small. When these weight values are really small, it reduces the effects of a lot of units in the hidden layers. Which makes the model behave more simply



