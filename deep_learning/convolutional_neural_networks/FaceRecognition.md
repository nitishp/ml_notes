# Face Recognition and Neural Style Transfer

## Face Recognition
* Two types of problems
  * Face verification: Does the image and the ID of the person match
  * Face recognition: Given the image of a person, output the ID of the person in the image
    * This is harder than the verification problem since errors can compound
* One Shot Learning
  * In these recognition problems, a traditional CNN would take in 1 image as an input, and have a softmax layer with a class for each person it wanted to recognize
    * This doesn't work well because you could have new people added to the set of predictions (someone new joins the company). In this case, you'd have to retrain your model
    * You need lots of training examples for each class in your training set. This isn't feasible to get
  * Instead what you'll have the network learn is something like a "Similarity" function between two images. So two images are considered similar if:
  $$
  d(img1, img2) \leq \tau
  $$
* Siamese networks (An application of oneshot learning)
  * Feed two images into a CONV network to get two flattened list of neurons. These will be the "representations" of the images
  * You compare the L2 norm of the representations of the real image with your reference image to compute the similarity score
  $$
  d(x^{(i)}, x^{(j)}) = ||f(x^{(i)}) - f(x^{(j)})||^2
  $$
  * You want this score to be low when the two images are the same person and big when the two images are of different people
  * For figuring out the parameters to generate the "representations", there's two loss functions you can use:
    * Tiplet Loss
      * For this, you use 3 images, an anchor (A), positive (P), negative (N). 
      * When training the dataset, you'll need to pick pairs of 3 images. And when picking images, you'll want to pick pairs where $d(A, P)$ and $d(A, N)$ are close to one another.
      * More formally, the loss function is defined as:
      $$
      Loss = max(d(A, P) - d(A, N) + \alpha, 0)
      $$
    * Straight binary classification
      * Put the two "representations" of images into a logistic regression unit to output if they're the same image
      * The model for computing the "representations" of each image will be the same with the same exact weights
      * The training set will be pairs of images
      * The formula for the logistic regression unit can be defined as something like:
      $$
      \hat{y} = \sigma(\sum_{i=1}^{k}w_k *|d(x^{(i)}, x^{(j)})| + b)
      $$
## Neural Style Transfer
* Given a content image (C), a style function (S), generate a generated image (G)
* Early layers of a CONV net detect simple features like an edge detector. Later layers automatically learn more complex features like a dog detector.
  * You can measure this by trying to see what images and inputs activate one particular hidden unit
* Training a neural style transfer algorithm
  * $J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)$
    * For $J_{content}(C, G)$
      * Pick a random layer $l$
      * Get the activations of layer $l$ for both $C$ and $G$. Compute something like:
        * $J_{content}(C, G) = \frac{1}{2}||a^{[l](C)}  - a^{[l][G]}|| ^ 2$

    * For $J_{style}(C, G)$
      * Across multiple layers $l$:
        * $J_{style}(C, G) = \sum_{l} \lambda^{[l]}J_{style}^{[l]} (S, G)$
      * $J_{style}^{[l]} (S, G) = \sum_{k} \sum_{k'} || G^{[l](S)} - G^{[l](G)}||_F$
        * $F$ indicates Forbenius norm
      * $G^{[l]}$ is the style matrix of size $(n_c^{[l]}, n_c^{[l]})$:
        * $G^{[l]}_{k, k'} = \sum_i \sum_j a^{[l]}_{i,j,k} a^{[l]}_{i,j,k'}$

  * Run gradient descent on a Generated output:
    * It starts off with a bunch of random pixel values
    * Each time the pixel values are updated with gradient of $J(G)$