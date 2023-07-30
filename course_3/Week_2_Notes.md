# Recommender Systems

- Running example throughout this set of notes: Recommending movies to users that they're likely to rate as 5 stars
- Some notation:
    - $n_u$ is the number of users
    - $n_m$ is the number of movies
    - $r(i, j)$ if user $j$ has rated movie $i$
    - $y^{(i, j)}$ user $j$ rating of movie $i$
    - $m^{(j)}$ number of movies rated by user $j$
    - $n$ number of features for a movie
- Assuming you have some set of features $x^1, ..., x^n$ that describe some traits about the movie, this prediction process becomes a lot like linear regression!
    - For each user $j$ you're trying to predict the rating they'd give a movie $i$, which looks like this:
        - $$ 
            rating = w^{(j)} \cdot x^{(i)} + b^{(j)}
          $$
    - And the cost function to minimize this rating is:
        $$
            J(w^{(1)}, b^{(1)}) = \frac{1}{2 m^{(j)}}\sum_{i:r(i,j)=1}^{n_m} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2 m^{(j)}}\sum_{k=1}^{n}(w_k^{(j)})^2
        $$
    - You end up wanting to basically make a linear regression for every single user and you can combine this information into one big summation cost (but in effect each user will have their own set of weights)
        - But you only sum over the movies the user has acutally rated
    - For finding all the $w^j$ and $b^j$ for all users, you can sum across all the users:
        $$
            J(w^{(1)}, b^{(1)}, ... , w^{n_u}, b^{n_u}) = \sum_{j=1}^{n_u} J(w^j, b^j)
        $$
- Collaborative filtering
    - Now assume you don't have features for each movie
    - Given some weights for each of the users $w^1, b^1, ..., w^{n_u}, b^{n_u}$, we can use the same cost function to come up with feature values $x^{(i)}$:
        $$
            J(x^{(i)}) = \frac{1}{2}\sum_{j:r(i,j)=1}^{n_u} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2}\sum_{k=1}^{n}(x_k^{(i)})^2
        $$
    - And similarly for computing every single $x^{(i)}$, just sum over every possible $i$:
        $$
            J(x^{1}, ..., x^{n_m}) = \sum_{i=1}^{n_m} J(x^{(i)})
        $$
    - We can actually create a combined cost function since both of these equations are summing over all pairs of (movie, user) ratings where $r(i,j) = 1$
    $$
        J(w,x,b) = \frac{1}{2} \sum_{(i,j):r(i,j) = 1} w^{(j)} \cdot (x^{(i)} + b^{(j)} - y^{(i,j)}) ^ 2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(w_k^{(j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
    $$

    - For minimizing the cost function, we can run gradient descent with partial derivities again:
    
    $$
        w_i^{(j)} = w_i^{(j)} - \alpha \frac{\partial J(w, b, x)}{\partial w_i^{(j)}}
    $$

    $$
        b^{(j)} = b^{(j)} - \alpha \frac{\partial J(w, b, x)}{\partial b^{(j)}}
    $$

    $$
        x_k^{(j)} = x_k^{(j)} - \alpha \frac{\partial J(w, b, x)}{\partial x_k^{(j)}}
    $$

- Binary labels for collaborative filtering:
    - Replace predicition function with the sigmoid function
        $$
            f_{(w,b,x)} (x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})
        $$
    - Where $g$ is the sigmoid activiation function
    - Cost function now becomes:
        $$
            L(f_{(w,b,x)} (x), y^{(i, j)}) = -y^{(i,j)} log(f_{(w,b,x)} (x)) - (1 - y^{(i,j)})log(1-f_{(w,b,x)}(x))
        $$

        $$
            J(w,b,x) = \sum_{(i,j):r(i,j)=1} L(f_{(w,b,x)}(x), y^{(i,j)})
        $$
