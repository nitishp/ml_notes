## Error Analysis

* Situation: You're not happy with the performance of your classifier
* You can conduct error analysis by:
  * Going through your dev set or real world set and identify mislabeled samples
  * Make a spreadsheet and go through each mislabeled example to try and categorize the errors
    * You can focus on concrete steps like getting more training data for the specific error categories
    * This can also give you an estimate of "how impactful" the problem is. You can use the percentage of error in the dev set as a representation of how inaccurate your classifier is

* Cleaning up incorrectly labeled data
  * Deep learning is quite resilient to small random errors in mislabeled training set data
    * It is far less resilient to "systemic errors". For example, classifying white dogs as cats
  * It might be worth doing a similar error "impact" analysis on your dev set to see how much correcting the wrong labels could improve your classifier
    * If you change the mislabeled examples in the dev set, definitely go through and do this on the test set as well
    * It might be okay to leave the training set as is.
* General tip:
  * Start with a quick and dirty implementation of the model
  * Use bias/variance tradeoff and error analysis to try and improve the model

## Mismatches Training and Dev/Test Set
* Using more training data from a different distribution might be okay
  * But it's really important to make sure that the dev/test sets are from the distribution you expect in production
  * Since it helps establish where your "target" is for your ML model
* Potential problems
  * Now that your training and dev sets come from different distributions, if there's a gap, it's hard to tell if it's a variance problem or a data mismatch problem
    * To detect this, include a training-dev set in addition to the dev/test set
    * This is the same distribution as the training data but is held out from training
  * If the error between the bayes error and training error is high -> High bias
  * If the error between the training error and the training-dev error is high -> High variance
  * If the error between the training-dev error and dev error is high -> Data mistmatch
* If there's a data mismatch problem, it'll be good to do error analysis, and then figure out how to increase more of those examples in your training set
  * One good way to do this would be with synthesized examples

## Learning from Multiple Tasks

* Transfer Learning
  * When to use
    * The two tasks have roughly the same input X (images or speech data)
    * You have a lot of data for Task A, and comparitvely less for Task B
    * Low level features from Task A could help for Task B
  * Rough process
    * Use pretrained model for Task A. Change the final neuron to output your desired data format (bool, multiple bools etc)
    * Train model on your training data
      * Whole model if you have a lot of training data
      * Just the neurons you added if you don't have that much data
* Multi-task learning
  * Train one neural network to give you multiple outputs. For example, does this image have a car? Does it have a stop sign? Does it have a pedestrian? etc.
  * This is different than multiclass classification because the answer for all the labels can be True
  * In practice,
    * Set up a neural network with multiple neurons for output of each question you want to answer
    * Used less often than transfer learning
  * When to use:
    * You think each of the outputs would benefit from learning the same low level data. Training one big neural net would be more efficient than having separate models for each output.
    * You have similar amounts of labeled data for each of the output. It's okay if there's some missing (just don't use those for the loss function)

## End-to-end Deep Learning