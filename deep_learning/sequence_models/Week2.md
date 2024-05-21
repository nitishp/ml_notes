## Word Embeddings

* In last week's RNN's, we represented each word as a 1-hot encoded vocabulary
  * Notation $O_{index}$ to indicate one-hot encoding
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