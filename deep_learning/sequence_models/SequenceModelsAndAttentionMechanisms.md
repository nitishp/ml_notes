# Sequence Models and Attention Mechanisms
* You can structure sequence models in multiple ways:
  * Encoder / Decoder:
    * Encoder and Decoder are both RNN
    * The hidden state from the last encoder is passed to the decoder
  * You can use a CNN to capture hidden state and pass into a decoder RNN
* In encoder-decoder model, a decoder is similar to a language model!
  * Decoder is trying to predict $P(y^{<1>}, ..., y^{<T_y>} | x)$
  * Decoder ideally wants to pick the $y^{<1>}, ..., y^{<T_y>}$ that maximizes this probability. And a greedy approach of picking the most optimal word at each step doesn't guarantee the optimal solution
  * You can use beam search to find this
    * Keep a parameter $B$ (beam width)
    * Algorithm:
      * Make $B$ picks for $P(y^{<1>} | x)$
        * Pass each $B$ option to RNN to pick $P(y^{<2>} | x, y^{<1>})$
          * $P(y^{<2>}, y^{<1>} | x) = P(y^{<2>} | x, y^{<1>}) * P(y^{<1>} | x)$
        * Pick the top $B$ values for this probability and continue
    * Large $B$ is more expensive to compute, small $B$ might not be accurate
    * To debug:
      * Go through incorrectly labeled examples in dev set, compute their probabilities and see if the probability picked was optimal. If not, then increase $B$

## Attention Mechanism
* Problem:
  * Current encoder / decoder model passes all encoder info directly into decoder with just one hidden state
  * Instead you can kind of stack the two RNNs on top of each other like so:
  ![Attention model](./attenion_model.png)
     * TODO Make this image better
  * This will let the model learn which specific parts of the input needs to be paid attention to for each predicted word
