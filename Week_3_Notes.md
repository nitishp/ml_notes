# Week 3 Notes
Classification
- Output values into a "class" or output category instead of predicting a continuous range of values
- Why won't linear regression work here?
  - You could in theory fit a best fit line and pick an arbitrary threshold, say 0.5, and say any predicted value below this gets the value of 0
  - This won't work if there's a strong outlier (since that changes the best fit line, even when it shouldn't change the prediction threshold)
  - It's easier to see with an example presented in lecture with the training slides (predicting tumor size vs if its malignant or not). If the threshold was say tumor size < .3  results in a negative prediction reasonably well, having a tumor of size 1000 throws off the best fit line to predict things incorrectly (it moves the threshold way to the right, but it shouldn't, since its a "good" outlier in this case)
