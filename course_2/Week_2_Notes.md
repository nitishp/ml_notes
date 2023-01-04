# Week 2

- Tensorflow
  - Use previous week's code to setup model
  - Call `model.compile(loss=BinaryCrossEntropy())`
    - BinaryCrossEntropy() is the same loss function as the logistic regression loss function
  - Call `model.fit(X,Y,epoch=<some_num>)`
    - Epochs control how many runs of gradient descent
    - This `fit` function even takes care of the complicated work of computing backpropogation or computing the partial derivates properly for the gradient descent update step
- Activation functions
  - Don't just have to be sigmoid, can also be:
    - ReLU: $g(z) = max(0, z)$
    - Linear: $g(z) = z$
  - In practice
    - Hidden layers: Almost always use ReLU
      - This makes gradient descent a bit faster in practice because the derivative term doesn't have multiple flat spots
      - **Don't** use linear because then it just results in linear regression everywhere
        - Linear algebra rule that a "linear function of a linear function is still a linear function"
    - Output layer:
      - Use sigmoid for binary classification
      - Use linear for regression
- Multi-class classification
  - Use softmax regression
    - Extension of logistic regression
    - Computation:
      - $$ z_j = w_j \cdot x_j $$
      - $$ a_j = P(y=j | x) = \frac {e^{z_j}} {\sum_{k=1}^{N} e^{z_k}}
      - Notice that computing $a_j$ is a function of all the other $z$ values. This is different when compared to sigmoid activations
    - Cost function
      - Loss is a piecewise function
        - $ Loss = -log(a_j)$ if $y = j$
      - Cost is an average of the loss over all the training examples
    - In tensorflow code:
      ```
      model = Sequential([
        Dense(units=2, activation='relu'),
        Dense(units=4, activation='relu'),
        Dense(units=3, activation='linear'),
      ])
      model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
      model.fit(X_train, y)
      z_out = model.predict(X_new)
      a_out = tf.nn.softmax(z_out) // Need to call activation since it's missing
      ```
      - Use `from_logits` for greater numerical precision (it just moves some numbers around)
      - You can always inspect individual layer weights and plot them to get a more intuitive sense of what each layer is doing
- Multi-label classification
  - Each example can have one or more labels. For example, a picture can have both a pedestrian and a car in it
  - You can just do this in the final output layer by having sigmoid activation functions for each of the neurons

- Adam algorithm
  - An alternative to gradient descent
  - The intuition behind this is that it's possible to dynamically adapt $\alpha$ based on the trajectory gradient descent is taking
    - If its heading in the same direction constantly, speed it up (increase $\alpha$)
    - If not, lower $\alpha$

- Convolutional neural networks
  - In contrast to the Dense neural networks, this neuron won't take in all of the previous inputs
  - If we had $x_1...x_{100}$ for example, we could have the first neuron take in $x_1...x_{20}$ and the second neuron take in $x_{11}...x_{30}$
  - Intuition:
    - You might want to do this to reduce the number of training examples needed
    - It also converges faster than a dense neural network
