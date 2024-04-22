# Foundations of Convoluational Neural Networks

## Convolutional Neural Networks
* In computer vision, we want to be able to use large images. But if we feed each pixel of the input as a parameter to the ML model, we'll have to train way too many parameters
* Convolution operator:
  * Notation is $*$ (same as multiplication)
  * Think of this like applying a "filter" to the matrix
  * Example:
  ![Convolution example](./convolution_example.png)
  * The example is a vertical edge detector (because of the filter)
    * You can flip the "Filter" matrix 90 degrees and get a horizontal edge detector
    * You can use deep learning to figure out 9 random numbers to learn to even detect 45 degree or 70 degree edges
* Padding
  * Notice that the output in the example is 4x4
  * If you have a n x n image, and an f x f filter, the result is a (n - f + 1, n - f + 1) image
  * To counteract this you can pad the input image all around to make sure it retains the same original size. This is called a "Same" convolution. 
  * If you do no padding, this is a "valid" convolution
  * You never add padding to the 3rd layer
* Strided convolutions
  * Instead of hopping over the input matrix one pixel at a time, you can take a stride of $s$ steps
  * This makes the final output matrix:
  $$
  (\lfloor \frac{n + 2p - f}{s} + 1 \rfloor, \lfloor \frac{n + 2p - f}{s} + 1 \rfloor)
  $$
* Convolutions over multiple dimensions
  * Normal images are (n, n, 3). You convolve this with a (f, f, 3) matrix. The 3's at the end have to be the same:
    * This gives a 2D matrix (called the feature map) with the following dimensions:
    $$
    (\lfloor \frac{n + 2p - f}{s} + 1 \rfloor, \lfloor \frac{n + 2p - f}{s} + 1 \rfloor)
    $$
    * If you apply multiple convolutions, you put them in different layers. So if applying multiple convolutions, you'll have an output matrix of the size:
    $$
    (\lfloor \frac{n + 2p - f}{s} + 1 \rfloor, \lfloor \frac{n + 2p - f}{s} + 1 \rfloor, n_{c'})
    $$
    * Where $n_{c'}$ is the number of convolution filters applied
* Single layer of a convolutional neural network
  * You can think of the filter in the convolution operator as part of $W^{[l]}$. More concretely, look at this example:
    ![Convolution Neural Nets](./convolution_neural_nets.png)
    * Number of parameters is $f^{[l]} \cdot f^{[l]} \cdot n_c^{[l - 1]}$
    * Note that you apply activation functions to the result of the convolution!
    * Multiple convolution filters apply add a new matrix in the 3rd dimension 
  * Notation
  ![Convolution Notation](./convolution_notation.png)
    * In this image if $l = 1$, then $n_c^{[l - 1]}$ is equal to the number of channels in the image
  * We will use backpropogation to learn $W^{[l]}$ (the values of the filter)
* Multiple layers of convolutional neural network
  * Called ConvNet for short
  * Apply multiple layers of these type of convolutions
    * Gradually the height and width of these go down as more layers are applied and the $n_c^{[l]}$ of each layer increases
    * "Unroll" the final layer and feed it into a logistic regression unit
* Pooling
  * Also splits up the image into $(f, f)$ chunks, but instead of doing convolution, it can either take the max value of this region (max pooling), or the average (average pooling)
  * It uses the same formula for determining the size of the output
  $$
    (\lfloor \frac{n + 2p - f}{s} + 1 \rfloor, \lfloor \frac{n + 2p - f}{s} + 1 \rfloor, n_{c'})
  $$
  * Keep the same number of layers in the third dimension
  * There is still backpropogation in pooling! But there's no parameters here so there's nothing to update
  * Why pooling
    * It helps reduce computation in the input since we shrink the input
* Fully Connected Layers
  * This is what we've seen before. Every neuron is connected to every other neuron
* Why Convolutions
  * It lowers the number of parameters needed to learn as compared to a fully connected layer between two matrices
  * It takes advantage of two things:
    * Parameter sharing: The weights learned for the filter (like for a vertical edge detector) are useful in any part of the image, either the top left or the bottom right
    * Sparsity of connections: When using convolutions, we only look at certain regions of the input when computing the value for a pixel. This is less information than a fully connected network


### Tensorflow Notes
* Sequential API
  * You can use `tf.keras.Sequential` to apply these types of layers sequentially to your neural network when building these models
  * They work similarly to Python Lists! 
  * Example:
  ```
  def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([            
            # YOUR CODE STARTS HERE
            tfl.ZeroPadding2D(padding=3, input_shape=(64,64,3)),
            tfl.Conv2D(32, 7),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(1, activation='sigmoid')
            # YOUR CODE ENDS HERE
        ])
    
    return model
  ```
* There's also the Functional API that lets you create a graph of layers.
  * The main benefit is it allows you to skip layers but I'm not sure when you would want to do this
  * They are both functionally equivalent
  * Example:
  ```
  def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    Z1 = tfl.Conv2D(8, 4, 1, padding="same")(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8,8), strides=(8,8), padding="same")(A1)
    Z2 = tfl.Conv2D(16, 2, 1, padding="same")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4,4), strides=(4,4), padding="same")(A2)
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(6, activation='softmax')(F)
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
  ```