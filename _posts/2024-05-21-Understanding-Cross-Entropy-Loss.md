---
Understanding Cross-Entropy Loss
---


### Introduction to Entropy

Entropy was firstly invented in Shannon's Information Theory, It measures the uncertainty or unpredictability in a system. Before the invention of Information Theory, people didn’t  how to  quantifies the amount of information required to describe the state of a system. You may ask: Is the ‘Uncertainty’ same as the uncertainty we use in our daily life, such as im uncertain if he’ll come to dinner. Its similar but not exact. In our daily life when we use ‘uncertainty’, we describe a state of being unsure or having doubts about something. It refers to a lack of confidence or clarity in our knowledge. However in the context of entropy and information thoery, ‘uncertainty’ has very specific and quantifiable meaning. It refers to the amount of information or surprise associated with outcome. Therefore, it is related to the probability distribution of a random variable. Lets take a look at an example.

Consider a fair coin toss. The outcome of a single toss is uncertain because there is an equal probability of getting heads or tails. The entropy of this system is high because each outcome carries an equal amount of information or surprise.
On the other hand, if we have a biased coin that always lands on heads, the outcome is certain, and the entropy of the system is low. There is no uncertainty or surprise associated with the outcome.

 The formula for entropy (H) of a random variable X with possible outcomes \( x_i \) and corresponding probabilities \( P(x_i) \) is:

$$
H(X)=−∑iP(xi)logP(xi)
$$

This formula calculates the expected amount of information (measured in bits) needed to encode the outcomes of X.

lets use the formulation to calculate the entropy in this case:

lets say if we toss 100, we got 56 heads(H) and 46 tails(T).

The probability of heads: P(H) : 56/100 = 0.56

The probability of tails: P(T): 46/100 = 0.

then the entropy would be:

$$
H(X) := -\sum_{x \in X} p(x) \log p(x) = - (0.56 * log(0.56) + 0.46 * log(0.46)) = 0.9837
$$

Lets say if the coin is unfair, if we toss 100, we got 82 heads(H) and 18 tails(T).

In the similar fashion, the entropy would be:

$$
H(X) := -\sum_{x \in X} p(x) \log p(x) = - (0.82 * log(0.82) + 0.18* log(0.18)) = 0.6801

$$

we can tell from this example that the later has lower entropy because the outcome is more certain given that we are more likely to get a heads.

### Entropy in Machine Learning

In the context of machine learning, entropy is used to measure the uncertainty in predictions. For a perfect model, the predicted probability distribution matches the actual distribution, resulting in low entropy. Conversely, higher entropy indicates a higher degree of uncertainty in the predictions. This loss function is widely used in classification problems because it quantifies how well the predicted probabilities match the true class labels.

### Cross-Entropy Loss

Cross-entropy loss builds upon the concept of entropy. It measures the difference between two probability distributions.

For binary classification:

$$
L = - \frac{1}{N} \sum_{j=1}^{N} \left[ y_j \log(p_j) + (1 - y_j) \log(1 - p_j) \right]
$$

For categorical classification:

$$
L = - \frac{1}{N} \sum_{j=1}^{N} \sum_{i=1}^{K} y_{ji} \log(p_{ji})
$$

Note: K refers to the number of classes

### Example of Cross-Entropy Loss

Consider a binary classification problem where we predict whether an email is spam (1) or not spam (0). Suppose we have the following true labels and predicted probabilities:

- True label: 1 (spam)
- Predicted probability: 0.8 (spam)

The cross-entropy loss for this single prediction is:

$$
H(P, Q) = -[1 \cdot \log(0.8) + 0 \cdot \log(0.2)] = -\log(0.8) \approx 0.22
$$

This value indicates the level of uncertainty in our prediction. If our model were perfect, the predicted probability would be 1, and the loss would be zero, indicating no uncertainty. Binary cross-entropy loss penalizes incorrect predictions, encouraging the model to output probabilities closer to the true labels.

### Cross-Entropy Loss in Categorical Classification

In multi-class classification problems, cross-entropy loss is extended to categorical cross-entropy loss. Here, the loss function compares the predicted probability distribution over multiple classes with the true distribution. For a classification problem with K classes, the categorical cross-entropy loss is:

For example, in a three-class classification problem (A, B, C) with true label B and predicted probabilities [0.1, 0.7, 0.2], the categorical cross-entropy loss is:

$$
L = -[0 \cdot \log(0.1) + 1 \cdot \log(0.7) + 0 \cdot \log(0.2)] = -\log(0.7) \approx 0.36
$$

### Conclusion

Cross-entropy loss is a powerful tool in machine learning, leveraging the concept of entropy from Shannon's Information Theory to measure the uncertainty in predictions. By penalizing incorrect predictions and rewarding correct ones, it guides models to improve their accuracy in both binary and categorical classification tasks. Understanding and effectively implementing cross-entropy loss is crucial for developing robust and reliable predictive models.
