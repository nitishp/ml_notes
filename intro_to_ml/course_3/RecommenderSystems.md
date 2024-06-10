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
        J(w,x,b) = \frac{1}{2} \sum_{(i,j):r(i,j) = 1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)}) ^ 2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(w_k^{(j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
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
- Mean normalization
    - Useful for cases of new users
        - By default, we'd just predict 0 for all movies which isn't too helpful
    - The intuition is that we'll calculate the mean rating ($\mu_i$) for each movie and adjust the prediction like so:
        $$
            f(x) = w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_i
        $$
        $$
            y^{(i,j)} = y^{(i,j)} - \mu_i
        $$
    - And the rest of the cost function remains the same
    - So in effect we'll end up predicting the mean rating for each movie for a brand new user
- Tensorflow can figure out how to calculate partial derivatives for you
    ```
    w = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        fwb = w*x + b # Tell tape how to caclculate f(x)
        costJ = (fwb - y) ** 2 # Tell tape how to calculate J
    
    [djDw] = tape.gradient(costJ, [w])
    w.assign_add(-alpha * djDw) # Add learning rate * partial to w

    ```
- Finding similar items
    - We can use the hidden features that are calculated in collaborative filtering to find this
    - Find the $k$ which minimizes this equation (here $x^{i}$ is the movie you care about):
        $$
            \sum_{l=1}^n (x_l^{(k)} - x_l^{(i)})^2
        $$
- Collaborative filtering drawbacks
    - Doesn't give an easy way to use side information (genre, demographics)
    - Suffers from a cold start problem (how do I score a new movie for example)

## Content-based filtering
- Takes features of movies $x_m$ and features of users $x_u$ and tries to use similarities to predict ratings of movies for a user
    - This is different than collaborative filtering where it uses the ratings users gave movies to try and figure out what you'd rate a movie
    - Collaborative filtering just needs user behavior data (what did the user rate movie $i$)
        - Content based filtering needs data about both item and user features
- Note that the dimensions of $x_m$ and $x_u$ could be very different. But we want to compute the dot product of two vectors $v_m^{(i)}$ for movie $i$ and $v_u^{(j)}$ for user $j$. These two dimensions should match. 
- Making $v_m$ and $v_u$ is actually just two neural networks
    - Input to $v_m$ network is $x_m$, and input to $v_u$ is $x_u$
    - The dot product of $v_m$ and $v_u$ is taken to get the prediction
- Cost function
    $$
        J = \sum_{(i,j):r(i,j)=1} (v_m^{(i)} \cdot v_u^{(j)} - y^{(i, j)})^2 + \text{NN regularization term}
    $$
- Scaling
    - The inference of the neural network can take a while
    - Lets break this down a bit into two steps
        - Retrieval: Get items that you want to score (not the entire catalog). Can be based on similarity scores to last 10 songs/movies listened to
        - Ranking: Run inference and compute predictions for all the items retrieved. Show the highest ones
- Code for combining the two networks
```
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)  
    ### END CODE HERE ###  
])

item_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)
    ### END CODE HERE ###  
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()

cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
```

## Principal Component Analysis
- Algorithm to reduce dimensionality of data to easily visualize it
    - Creates a derived axis to help still keep the dimensionality low
    - Project each point onto this new axis and get the new value
    - New axis is called "principal component" that leads to max variance in the data
    - Each principal component is perpendicular to the original axis
- Implementation details
    - Run feature scaling and mean normalization for features
    - To project:
        - Take original coordinates and dot product it with the unit vector in the direction
        - "Reconstruction" is not exact, but can be used to approximate the dimensions
            - Take value and multiply it by the unit vector of the principal component axis
- Code
```
pca_1 = PCA(n_components=1) # num dimension
pca_1.fix(X)
pca_1.explained_variance_ratio_ # Variance difference between applying and not applying PCA
X_trans_1 = pca_1.transform(X)
```

