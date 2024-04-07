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
    * More formally, this reduces the covariate shift effect for each neural net layer:
      * This happens when the underlying dataset shifts slightly, causing the output of the layer to be incorrect
      * Batch normalization forces the mean and variance of the layer to be 0 and 1, so it coerces these values to be in a similar range
    * It also has the slight effect of regularization when used with mini-batch gradient descent, since the mean and variance are calculated per mini-batch. So they'll never be fully accurate, thus adding some noise
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
* During test time, you'll want to use the same formulas, but you need to use $\mu$ and $\sigma^2$ over the entire training set:
  * You can do this using exponentially weighted averages for each mini batch

## Multi-class Classification

* Called: Softmax regression
  * $C$ = number of classes
* The final layer of the neural network is the softmax layer
  * It has $C$ number of neurons
* To compute the final output of the softmax layer, the activation is a little different:
  * Compute $z^{[l]}$ as normal
  * Compute:
  $$
  t = e^{z^{[l]}}
  $$
  $$
  a^{[l]} = \frac{t}{\sum_{i=1}^{C}t_i}
  $$
  * Pick the highest value
* Cost function:
$$
Loss(\hat{y}, y) = -\sum_{j=1}^{C}y_jlog(\hat{y}_j)
$$
$$
J = \frac{1}{m} \sum_{i=1}^{m}Loss(\hat{y}^{(i)}, y^{(i)})
$$
* Computing back prop is the same the only difference is this equation:
$$
dz^{[L]} = \hat{y} - y
$$


## Programming Frameworks
* Tensorflow Notes
```
w = tf.Variable(0, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(0.1) #0.1 = learning rate

def train_step():
  with tf.GradientTape() as tape:
    cost = w ** 2 - 10 * w + 25
  trainable_variables = [w]
  grads = tape.gradient(cost, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))

for i in range(1000):
  train_step()
```

```
w = tf.Variable(0, dtype=tf.float32)
x = np.array([1.0, -10.0, 25.0])
optimizer = tf.keras.optimizers.Adam(0.1) #0.1 = learning rate

def training(x, w, optimizer):
  def cost_fn():
    return x[0] * w ** 2 + x[1] * w + x[2]
  for i in range(1000):
    optimizer.minimize(cost_fn, [w])
  
  return w

w = training(x, w, optimizer)
```

* TensorFlow datasets are stored as TensorFlow datasets
  * These only provide access to iterators (no direct access to elements)
  * `tf.data.Dataset.from_tensor_slices(<np_array>)`
  * Call `.map` to transform the TensorFlow dataset
* With tensorflow, all you need to do is:
  * Define the forward prop
  * Define the cost function
  * Gradients and backprop is taken care of for you

General workflow:
```
# Initialize params
# Pick optimizer
for epoch in range(num_epochs):
  with tf.GradientTape() as tape:
    # Do forward prop
    # Compute loss
  
  trainable_variables = [W1, b1, W2, b2, ..., Wn, bn]
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
```