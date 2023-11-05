# Unsupervised Learning

## Clustering Algorithm
- Used to group similar things together
- K-means algorithm
    - Start with some pre-defined $k$ clusters. Intuitively, there are two important steps:
        - Assign each data point to a cluster (based on whichever one its closest to)
        - Recompute the cluster centroid based on the average location of all the points
            - If a cluster has no points assigned to it, it's common to just delete it
        - Keep repeating this until no new points are assigned to different ones or the cluster centroids don't change
    - Cost function
        - The cost function normally used for k-means is:
            $$ 
            J = \frac{1}{m}\sum \limits_{i = 0}^{m-1} ||x^(i) - \mu_{c^{x^i}}|| ^2
            $$
            - This is just the average distance between each data point and the cluster its' assigned to
        - Intuitively, when following the k-means algorithm, both steps help to minimize this cost function 
            - In the first step, we choose to assign the point to the closest cluster centroid
            - In the second step, we choose to minimize the distance between each point and the cluster centroid it's assigned to
        - This cost function should never go up on any iteration of the code
    - Random initialization
        - It's common to pick $k$ random points as the initial $\mu_k$ points
        - k-means does have the potential to run into local optima
            - The way to get around this is to have multiple k-means runs (say 50-1000) and pick the k-means run that gives you the lowest $J$ value
    - How to pick the right $k$
        - A lot of times, it really is ambiguous
        - A common approach is to look at the downstream results of picking this value of $k$ (does it lead to less cost overall in your application for example)

## Anomaly Detection
- Used to detect unusual (anomalous) events
- Rough idea: You compute the probability ($p$) of an event like the new event $x_{test}$ happening. If $p(x_{test}) < \epsilon$, then you've detected an unusual event
- Gaussian distribution
    - $$
        p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{\frac{-(x - \mu)^2}{2\sigma^2}}
      $$
    - To compute mean and standard deviation
        - $$
            \mu = \frac{1}{m} \sum \limits_{i=0}^{m} x^{(i)}
            $$
        - $$
            \sigma^2 = \frac{1}{m} \sum \limits_{i=0}^{m} (x^{(i)} - \mu)^2
          $$
    - For multiple variables:
        - 
        $$
         p(x) = p(x_1;\mu_1, \sigma_1^2) * p(x_2;\mu_2, \sigma_2^2) * ... * p(x_n;\mu_n, \sigma_n^2)
        $$
- Anomaly detection in practice
    - You can use previous data to tune your anomaly detection algorithm
    - If you have 10,000 examples, split it into 6k training set (which is just all the normal / unanomalous events), 2k cross validation set (used to tune $\epsilon$), 2k test set (used for reporting errors)
- Anomaly detection vs supervised learning
    - When to pick anomaly detection 
        - If the number of anomalies in your training set is really small (0-20)
        - If you expect new anomalies that are completely different from the anomalies in your training set (an example is financial fraud where new types of fraud come up every few months)
- Try to make your features gaussian (since that's what this anomaly detection is based off of). You can do this through just modifying existing features (take the log of it or take the exponent of it), it's a pretty trial and error process
- In general, if you find that there are certain anomalies that your algorithm is not catching, it's worth taking those anomalies and analyzing them. Are there new features that could be added that would make it easier to catch this anomaly? Does the threshold need to be adjusted? etc.