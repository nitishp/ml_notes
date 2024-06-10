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
* ResNets
  * The core idea is to have "Skip connections". We feed the output of a layer $a^{[l]}$ two layers so that the output of something two layers ahead becomes:
  $$
  a^{[l + 2]} = g(z^{[l + 2]} + a^{[l]})
  $$
  * Performance of very deep plain neural networks tends to get worse on the training set as you add more layers. There gets to be too many parameters to train
  * Since $z^{[l + 2]}$ is just a product of $W^{[l + 1]}$, this allows the ResNet network to learn the identity function by setting $W^{[l + 1]}$ equal to 0. So it's easier to skip these additional layers if they're useless
  * A lot of times in ResNets, the number of dimensions is the same between layers, but if it's not, you can multiply $a^{[l]}$ by some matrix to get the dimensions to match. This is implemented in the "skip connections" path
* Inception Network
  * 1x1 convolution network
    * Used to shrink the number of channels ($n_c$) in the output
    * You apply ReLU after this, so it can also be used to apply nonlinearity to the output
    * They can be used to shrink down the number of computations that need to be done to apply a higher fxf computation
  * Core idea:
    * Instead of picking the different filter sizes for layer of the NN. Just try them all and stack the results in the output. So the output can be the result of a 5x5 conv layer, a 3x3 conv layer and a 1x1 conv layer. All the conv layers have "same" convolutions
      * This is called a Inception Module
* MobileNet
  * MobileNet V1 Architecture 
    * Depthwise separable convolution
      * Depthwise convolution - similar to normal convolution but the key difference is that you do multiplications on a per channel basis. So the filter is not a cube, but a plane instead
      * Pointwise convolution - Use a 1x1 convolution with a different number of filters to get the same output dimensions you want
    * This involves much less computation for forward prop compared to normal convolutions
  * MobileNet V2 Architecture (2019)
    * Very similar to v1, but the main difference is for convolution, it uses an expansion-depthwise-projection type of convolution:
      * Expansion - a 1x1 convolution with a higher number of filters to increase the number of channels
      * Depthwise convoluion - same as above
      * Projection convolution - same as Pointwise convolution
    * The main benefit of the expansion convolution is to be able to learn a more complicated function before having it shrink down again
    * It uses a final GlobalAveragePooling2D to reduce the output size

* Transfer learning
  * A good approach for a lot of computer vision tasks since a lot of these networks have been trained for weeks!
  * Depending on your dataset size, you can download someone else's network and weights and "freeze" the parameters. Then change the last softmax layer to output the classes that you care about
    * If you have more data, you can choose to freeze only certain layers
    * Lots of deep learning frameworks have the ability to freeze layers
  * When using a pre-trained model, it's best practice to use the same normalizations if the model did that too
* Data augmentation
  * A lot of computer vision tasks almost always never have enough data. It's common to use techniques like cropping, rotating, reflecting etc to get more data
  * This can lead to potential shifts in the data distribution between training / dev / test sets
    * Use train-dev sets to figure out if you're overfitting the model itself, or if you're data is too mismatched!