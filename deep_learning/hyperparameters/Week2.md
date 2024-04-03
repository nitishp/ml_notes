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
    * Called stochastic gradient descent
    * Will oscillate a lot or diverge (you can pick a lower learning rate)
    * Will be really slow since you lose vectorization benefits
    * Won't hit exact minima but can get close (you can pick closeness with a lower learning rate)
  * Picking middle values:
    * Try and pick a value of the power of 2
    * Size depends on application, but in general it's best to make sure that all the training data for a batch can fit on the CPU / GPU
* Exponentially weighted average
  * This isn't a formal average
  $$
  V_t = \beta V_{t - 1} + (1 - \beta)\theta_t
  $$
  * As a rule: It computes average over approximately $\frac{1}{1 - \beta}$ time units
  * Bias correction
    * Used to help address the cold start problem with exponentially moving algorithms
      * For computing $V_0$ there is no $V_{-1}$ so the value becomes really small, just $(1 - \beta)\theta_t$
    * To account for this, use the following formula:
    $$
    \frac{V_t}{1 - \beta^t}
    $$
* Gradient descent with momentum
  * Compute exponentially weighted average of your gradients, and use the gradient to update weights instead
  * Works for both batch gradient descent and mini-batch gradient descent
    * It helps smooth out oscillations when traversing to the minimum
* RMSProp
  * Similar to gradient descent with momentum
  * For each iteration of gradient descent:
  ```
  Compute dW, db on current mini-batch
  s_dw = beta * s_dw + (1 - beta) dW ^ 2
  s_db = beta * s_db + (1 - beta) db ^ 2

  W = W - alpha * (dW / sqrt(s_dw)
  b = b - alpha * (db / sqrt(s_db))
  ```
* Adam optimization algorithm
  * Combines both RMSProp and gradient descent with momentum
  * For each iteration of gradient descent / mini-batch gradient descent:
  ```
  Compute dW, db on current mini-batch
  v_dw = beta_1 * v_dw + (1 - beta_1) * v_dw
  v_db = beta_1 * v_db + (1 - beta_1) * v_db
  s_dw = beta_2 * s_dw + (1 - beta_2) * dW ^ 2
  s_db = beta_2 * s_db + (1 - beta_2) * db ^ 2

  v_dw_corrected = v_dw / (1 - (beta_1 ^ iteration_count))
  v_db_corrected = v_db / (1 - (beta_1 ^ iteration_count))
  s_dw_corrected = s_dw / (1 - (beta_1 ^ iteration_count))
  s_dw_corrected = s_dw / (1 - (beta_1 ^ iteration_count))

  W = W - alpha * (v_dw_corrected / sqrt(s_dw_corrected + epsilon)
  b = b - alpha * (v_db_correct / sqrt(s_db_corrected + epsilon))
  ```
  * Here `epsilon` is just a really small number to prevent the denominator from being 0
* Learning rate decay
  * During initial steps of learning you take bigger steps of gradient descent. As iterations continue, you take smaller steps.
    * i.e. $\alpha$ goes down over the number of iterations
    * Concrete formula:
    $$
    \alpha = \frac{1}{1 + decayRate \cdot epochNumber} \alpha_0
    $$
* Local optima problems
  * There's no real global minimum in very high dimensional spaces. But there can be a lot of saddle points
  * This has the issue where your gradients become really small (they're in a "flat region") before switching to a place where they lower