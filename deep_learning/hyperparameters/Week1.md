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

* Dropout Regularization (aka "Inverted Dropout")
  * During training
    * High level: Choose to randomly drop units in a hidden layer
    * Specify a `keep_prob` which is the probability of keeping a unit. This is implemented by:
      * Randomly "dropping" some of the output neurons in the activation of each layer. "Dropping" here means just setting them to 0
      * Scaling the output of each activation by dividing it by `keep_prob`. This helps keep the expected value of the output the same
    * The hidden units dropped changes for each training example
  * During test time
    * Don't do any dropping and use evaluation as you normally do. The scaling by `keep_prob` is supposed to take care of this
  * Intuition:
    * Removing nodes randomly leads to a simpler network
    * If I'm a node in a neural network with dropout:
      * One of my inputs could randomly disappear
      * It makes sense to spread out my weights to other inputs since the one I normally rely on could disappear
    * You can change `keep_prob` for each layer in a neural network
* Other high variance fixes:
  * Data augmentation techniques: Slightly modify your examples to add a little bit of variance. This gets you more data
  * Early stopping: Plot your cost of both the training set and the dev-set over the number of iterations. Eventually, it'll hit a point where the they start to diverge, and you want to pick that!
    * The downside here is it couples training the model with not overfitting. So it does make it more complicated to reason about.


