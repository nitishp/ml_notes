## Hyperparamter Tuning
* There's a lot of hyperparameters when training a model. Some examples:
  * $\alpha$, $\beta$, # of hidden units, # of layers etc
* In order to pick good ones, it's an empirical process
* Try to pick random values and see what works best:
  * Don't use grid search, since one hyperparameter could be really important, but you'll only search with a limited set of values
  * It's not always sampling randomly between two numbers, it might make sense to do sampling over a log scale:
    * For values like $\beta$ (the exponentially weighted average term), they work on an exponential scale:
      * There's not a huge difference between 0.9 to 0.00005. Both of these are considered to average over the last 10 samples
      * There's a big difference between 0.999 and 0.9995. This goes from averaging over the last 1000 samples to the last 2000 samples
    * To account for this, you can do the following:
      ```
      tmp = np.random.rand() * <some_num>
      random_sample_val = 10 ** tmp
      ```

## Batch Normalization
* High level idea:
  * We normalize inputs to make gradient descent faster
  * We can do the same thing for the activations of the output of each hidden layer
* In practice, the normalization is not done at $a^{[l](i)}$ but at $z^{[l](i)}$
* The way this works is:
  * For a bunch of activations: $z^{[l](1)},z^{[l](2)},...,z^{[l](m)}$
  * Compute the following:
  $$
  \mu^{[l]} = \sum_{i=1}^{m}\frac{z^{[l](i)}}{m}
  $$
  $$
  \sigma^2 = \frac{1}{m}\sum_{i=1}^m (z^{[l](i)} - \mu)^2
  $$
  $$
  z_{norm}^{[l](i)} = \frac{z^{[l](i)} - \mu^{[l]}}{\sqrt{\sigma^2 + \epsilon}}
  $$
  $$
  \tilde{z}^{[l](i)} = \gamma z_{norm}^{[l](i)} + \beta
  $$
* You then feed in the $\tilde{z}^{[l]}$ to the activation functions to get the inputs of the next layer
* Here $\gamma$ and $\beta$ are learnable parameters of the model:
  * They're useful because you might not want the values to just be between -1 and 1
  * $\beta^{[l]}$ actually ends up replacing $b^{[l]}$ as one of the parameters that the model needs to learn
* Batch norm applies to each mini-batch, so the sum and variances are calculated on a per mini-batch basis
* In terms of pseudocode:
```
for t=1,...,num_mini_batches
  compute_forward_prop with batch norm
    * Using the formulas above for each activation
  compute_backprop
    * learn dW, dgamma, dbeta
  update_parameters
    * dW = dW - alpha * dW
    * dgamma = dgamma - alpha * dgamma
    * dbeta = dbeta - alpha * dbeta
```

## Multi-class Classification

## Programming Frameworks