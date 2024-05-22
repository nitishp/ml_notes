## Word Embeddings

* In previous RNN's, we represented each word as a one-hot encoded vocabulary
  * Notation is $O_{index}$ to indicate one-hot encoding
* You can instead build word embeddings. 
  * Notation is $e_{index}$
  * These learn "features" of the word. It's helpful for finding relationships between words
    * It helps algorithms learn analogies, since they can find similar words
    * If you wanted to programmatically find a word with a similar relationship as king->queen. And the word you're given is man
      * You can concretely do this by finding the max $index$ where the $sim(e_{index}, e_{man}) \approx sim(e_{king}, e_{queen})$
      * You can use the cosine similarity or Euclidian distance for this
  * You can use a different algorithm to learn word embeddings and use the embedded words to train your RNN. This can yield better performance as the algorithm can learn relationships between words
  * Embedding matrix ($E$) is the matrix that holds all embedding values for each word
    * $E * O_{index} = e_{index}$
* Learning Word Emvedding matrix ($E$):
  * Trains on a large corpus of text
  * Two variants:
    * Word2Vec
      * Build a supervised learning model modeled like so:
        * $O_c \rightarrow E \rightarrow e_c \rightarrow softmax \rightarrow \hat{y} $
        * The $x$ and $y$ here are one hot vectors over the entire vocabulary where it's 1 for the given word
        * We're using this problem to just learn $E$ and $e_c$
      * Two variants within this:
        * Bag of words
          * Same model but uses a set number of context words before the target
        * Skip-gram model
          * Uses just one context word within some range of the target word
      * One issue with computing this model is that it can be slow since the number of vocabulary words is very large, so computing the $softmax$ can be slow
    * Negative sampling
      * Similar to skip-gram model, but randomly just pick one word in context and target pair and set it to 1. Use the other words within the context window and set the context-target pair to be 0
      * But changes the model to be:
        * $O_c \rightarrow E \rightarrow e_c \rightarrow logistic$
        * There are vocab_size * $logistic$ units
  * GloVe
    * Different algorithm to compute word embeddings
    * Use $X_{ij}$ which is a matrix that shows the # times $j$ appears in the context of $i$
    * Minimize:
    $$
    \sum_{i=1}^{10,000}\sum_{j=1}^{10,000} f(X_{ij})(\theta_i^Te_j + b_i + b_j - log(X_{ij})) ^ 2
    $$
      * Can just use gradient descent
    * Final word embedding:
      * $e_w = \frac{\theta_w + e_w}{2}$
* Debiasing word embeddings
  * You can manually tweak word embeddings based on bias in the training data
  * This can help fix any learned bias associated like woman = homemaker