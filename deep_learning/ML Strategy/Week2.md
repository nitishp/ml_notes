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
    * It might be okay to leave the training set as is. We'll learn more about this later in the week (TODO)
* General tip:
  * Start with a quick and dirty implementation of the model
  * Use bias/variance tradeoff and error analysis to try and improve the model