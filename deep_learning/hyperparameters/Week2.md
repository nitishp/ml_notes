## Optimization Algorithms

* Picking better cost optimization algorithms can help train models faster
* Mini-batch gradient descent
  * Works similar to gradient descent, but breaks the massive training set into smaller "batches" (say of size 1000)
  * Gradient descent works as normal on these smaller batches
  * Notation: $X^{\{t\}}$ refers to the batch $t$
  * Code
  ```
  for i = 1 to num_iterations:
    for t = 1 to num_mini_batches:
      Y_pred, cache = forward_prop(X_t, W, b)
      J_t = cost(Y_pred, Y_t)

      ...backward_prop_code...

      W -= alpha * dW
      b -= alpha * db
  ```
* Mini-batch gradient descent doesn't lower the cost on every single epoch of the training set. It will trend lower but not every single iteration. It's because it's like on each epoch you're training on a different training set
* Picking different mini-batch sizes:
  * If == m:
    * Same as batch gradient descent
    * Really slow for very large training sets
  * If == 1:
    * Will oscillate a lot or diverge (you can pick a lower learning rate)
    * Will be really slow since you lose vectorization benefits
    * Won't hit exact minima but can get close (you can pick closeness with a lower learning rate)
  * Picking middle values:
    * Try and pick a value of the power of 2
    * Size depends on application, but in general it's best to make sure that all the training data for a batch can fit on the CPU / GPU