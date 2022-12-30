# Week 1

- Neural Networks
  - Loosely based on biological neurons (but not anymore since we have very little idea about how the brain actually works)
  - There's lots of interconnected neurons that take in a bunch of numbers and output a number
    - The output of neurons are connected to the input of other neurons (except for way at the end)
  - One way to think of neural networks is just nested logistic regression, it takes a bunch of features, outputs values for a hidden layer and then feeds the hidden layer output to another hidden layer or the output neuron
    - Refer to the slide for more info
- Terminology
  - Activation is used interchangably with $a$
  - $$a = f_{w,b}(X) = \frac {1} {1 + e^{-(wx + b)}}$$
  - Hidden layer: The layer of neurons in between the input and output
    - Called "Hidden" because when we have our training data, we don't know what the outputs here should be $y$ will just refer to the final output data
  - $a^{[2]}$ refers to the output of the 2nd layer

- It's common for hidden layers to be connected all the input features
- Picking how many hidden layers and how many neurons each hidden layer should have is one of the fundamental things to engineer
- Image recognition: Inputs to neural network are just all the pixel values
  - i.e. the 2D matrix is unrolled into one giant vector of say 1 million pixel values for a 1000x1000 image
- Output of one hidden layer gets fed into another layer, i.e.:
  - $a^{[1]} = g(w^{[1]} \cdot x + b^{[1]})$
  - $a^{[2]} = g(w^{[2]} \cdot a^{[1]} + b^{[2]})$
  - More generically:
    - $x = a^{[0]}$
    - $a_j^{[l]} = g(w_j^{[l]} \cdot a^{[l - 1]} + b^{[l]})$
  - This is called "Forward propogation"

- Tensorflow
  - Key code:
    ```
    model = Sequential([
      Dense(units=2, activation='sigmoid')
      Dense(units=4, activation='sigmoid')
    ])
    model.compile() // More on this next week
    model.fit(X_train, y) // More on this next week
    y_hat = model.predict(X_new)
    ```
  - By default Tensorflow operates on 2D arrays for carrying out large computations efficiently
    - So by default make the input `x` a 2D array
  - When you see a `tf.Tensor` datatype, just think of it as a 2D array

- When thinking of $W$ matrix, think of it grouped by columns, i.e a 2x3 matrix would correspond to 3 neurons each with 2 features
