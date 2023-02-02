# Week 4

- Decision Trees
    - Forward Prediction works like a Tree
        - Start at the root note and choose to go to any of the children
        - Here's an example:
            ![Decision Tree Example](./decision_tree_example.png)
    - Learning Questions
        - How do you decide which feature to split on?
            - Intuitively you want to pick the node that will most cleanly split your data, so that each side is classified differently
        - When do you decide to stop splitting?
            - Could be when 100% of examples are of one category
            - Could be once the tree hits a certain depth, so as to avoid being too large and thus create overfitting

