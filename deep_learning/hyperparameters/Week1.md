* Applied ML is a highly iterative process
  * Number of layers, number of hidden units, learning rates are all things you have to really experiment with
  * Intuitions from one domain (NLP, Computer Vision) don't translate to other domains
  * No one really picks the best hyperparameters the first time
* Ratios for splitting into train/dev/test sets are very different based on how much data you have
  * Definitions
    * Training set: Used to train model
    * Dev set: Used to evaluate different models
    * Test set: Used for final reporting of error rates
  * If you have a lot of data (say millions of examples), its okay to have dev and test sets be a lot smaller (say 1% of the entire dataset)
  * In general, try to keep the dev/test sets as close as possible to production data to have a better sense of model accuracy in the real world

* High bias = underfitting the data
  * It's "Biased" a lot by the model
  * High training set error, high dev set error
* High variance = overfitting the data
  * It "varies" a lot with the training data
  * Low training set error, high dev set error
* It's possible to have both high bias and high variance. It means your model is bad and it's overfitting the training data
