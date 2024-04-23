## Case Studies

* Classic Network
  * Le-Net 5 (built in 1998 by Yann LeCun)
    * Used for recognizing grayscale handwritten digits
    * Of the form:
    ```
    CONV_2D with 6 filters and filter size of 5
    AVG_POOL with filter = 2 and stride = 2
    CONV_2D with 16 filters and filter size of 5
    AVG_POOL with filter = 2 and stride = 2
    FULLY_CONNECTED layer of 120 neurons
    FULLY_CONNECTED layer of 84 neurons
    SOFTMAX layer with 10 output
    ```
    * As you go deeper in the layer, the $n_H$ and $n_W$ shrink while the $n_C$ increases
    * Trained around 60k parameters
  * AlexNet (built in 2012)
    * Of the form:
    ```
    CONV_2D with 96 filters, f = 11, s = 4
    MAX_POOL with filter = 3 and stride = 2
    CONV_2D with 256 filters and "same" filters
    MAX_POOL with filter = 3 and stride = 2
    CONV_2D with 384 filters and "same" filters
    CONV_2D with 384 filters and "same" filters
    CONV_2D with 256 filters and "same" filters
    MAX_POOL with filter = 3 and stride = 2
    UNROLL prev layer
    FULLY_CONNECTED layer of 4096 neurons
    FULLY_CONNECTED layer of 4096 neurons
    SOFTMAX layer with 1000 output
    ```
    * Similar to LeNet but much deeper
    * Trained around 60M parameters
  * VGG-16 (built in 2015)
    * Trains around 150M parameters
    * Similar ideas to AlexNet, but has more layers
    * Key idea: Each CONV layer is a 3x3 "same" filter, and each MAX_POOL layer is 2x2 with a stride of 2
      * Roughly the number of filters in each CONV layer doubles after each MAX_POOL layer
      * There are also multiple CONV layers in a row