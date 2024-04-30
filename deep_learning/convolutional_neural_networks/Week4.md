# Face Recognition and Neural Style Transfer

## Face Recognition
* Two types of problems
  * Face verification: Does the image and the ID of the person match
  * Face recognition: Given the image of a person, output the ID of the person in the image
    * This is harder than the verification problem since errors can compound
* One Shot Learning
  * In these recognition problems, you often just have 1 image of each person. So traditional CNN don't work as well
  * Instead what you'll have the network learn is something like a "Similarity" function between two images. So if:
  $$
  d(img1, img2) \leq \tau
  $$
  * Then the two images are considered similar
  * Siamese networks do this
    * Feed the image into a CONV network to get a flattened list of neurons. These will be the "representation" of the image
    * You compare the L2 norm of the representations of the image to compute the similarity score
    $$
    d(x^{(i)}, x^{(j)}) = ||f(x^{(i)}) - f(x^{(j)})||^2
    $$
    * You want this score to be low when the two images are the same and big when the two images are different