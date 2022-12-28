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

- It's common for hidden layers to be connected all the input features
- Picking how many hidden layers and how many neurons each hidden layer should have is one of the fundamental things to engineer
- Image recognition: Inputs to neural network are just all the pixel values
  - i.e. the 2D matrix is unrolled into one giant vector of say 1 million pixel values for a 1000x1000 image
