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

