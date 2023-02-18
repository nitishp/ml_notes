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