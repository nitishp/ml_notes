# Transformer Networks

* Conceptually very similar to attention model
  * Does some parallel processing CNN style
* Transfer model can be better than computing static word embeddings since it learns directly from the sentence itself
* It helps compute better hidden states for
* Self attention
  * For each word compute $A(q,k,v)$. This is the attention-based representation of the word
    * $x^{<i>} \rightarrow A(q,k,v) \rightarrow A^{<i>}$
    * $A(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$
    * $Q,K,V$ are parameters the learning algorithm learns
* Multi-headed attention
  * Think of it as a big "for-loop" over self attention
  * Another good way to think about this is kind of like convolutional layer filters (how there are multiple)
  * $Multihead(Q,K,V) = concat(head_1,...,head_n)W^o$
* You can also encode positional info in the attention model. This is useful since positional information can be very valuable when reconstructing sentences
  * It's added directly to each $x^{<i>}$ before being passed into $A$
