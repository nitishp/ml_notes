## Shallow Neural Network

* Notation
  * $W^{[1]}$ and $W^{[2]}$ refer to the weights of the network at layer 1 and layer 2
  * $a^{[l]}_{i}$ represents the i-th element in the activation of layer l
  * $W_i^{[l]}$ represents the weights for the i-th element in layer l
  * Input layer: Inputs to the network
  * Final layer / output layer: Responsible for generating output $\hat{y}$
  * Middle layers are called hidden layers
* Basic idea: Taking logistic network and repeating it twice

* Computing Neural Network output for single example
 $$ Z^{[1]} =
  \begin{bmatrix}
  ... & W_{1}^{[1] T} & ... \\
  ... & W_{2}^{[1] T} & ... \\
  ... & W_{3}^{[1] T} & ... \\
  ... & W_{4}^{[1] T} & ... \\
  \end{bmatrix}

  \begin{bmatrix}
  x_1 \\
  x_2 \\
  x_3 \\
  \end{bmatrix}

  +

  \begin{bmatrix}
  b_1^{[1]} \\
  b_2^{[1]} \\
  b_3^{[1]} \\
  b_4^{[1]} \\
  \end{bmatrix}
 $$
 $$
  A^{[1]} = \sigma{(Z^{[1]})}
 $$

In this example:
* The dimensions of $X$ are $(3,1)$, where the $1$ represents 1 training example
* The dimensions of $W^T$ are $(4, 3)$ because there are 4 outputs and 3 inputs, and similarly the dimensions of $b$ are $(4, 1)$
* The dimeions of $Z^{[1]}$ and $A^{[1]}$ are both $(4, 1)$ because of the calculation

A different way of thinking of this is the dimensions of each $W^{[l]}$ will be (num outputs, num inputs)

Working across multiple examples:
* Recall that $X$ here is a matrix of $(n_x, m)$ where $n_x$ represents the number of features and $m$ represents the number of training examples
* We can calculate the activation of layer l as $A^{[l]} = \sigma{(W^{[l]} X + b^{[l]})}$
  * This will output $A^{[l]}$ as a $(n_h, m)$ matrix where $n_h$ represents the number of hidden units
  * You can generalize $X$ to be $A^{[l - 1]}$ since $X = A^{[0]}$

Activation functions
  * Sigmoid: Useful for 0 or 1 classification in the output layer
  * Tanh: Similar to sigmoid, but ranges between -1 and 1
  * ReLU: Formula is $max(0, z)$ and is commonly used in the activation function of hidden layers

Why use non-linear activation functions
* So you can use a linear activation function in the output layer if your problem has a continuous space of values
* But in general, the composition of two linear functions is just a linear function. Consider the following:

$$
a^{[1]} = z^{[1]} = W^{[1]}X + b^{[1]}
$$

$$
a^{[2]} = z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}
$$

$$
a^{[2]} = W^{[2]}W^{[1]}X + W^{[2]}b^{[1]} + b^{[2]} = W'X + b'
$$

Backpropogation intuition

![Backprop diagram](./backprop.png)

Random Initialization
* What happens if you intialize all of $W$ to be 0?
  * It ends up that all of the individual hidden units end up computing similar $W$ values and they all end up behaving similarly
  * To offset this, randomly initialize them to small values
* Why do you initialize to small values
  * If you're using functions like sigmoid or tanh, initializing to small values helps the algorithm converge faster
