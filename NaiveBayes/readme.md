# Naive Bayes

## Bayes Theorem
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

## ML Case
$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

Dengan $X = (x_1, x_2, x_3, ..., x_n)$

## Class Conditional Probability $P(x_i|y)$
$$Posteriors = P(x_i|y) = \frac{1}{\sqrt{2 \pi \sigma_y^2}} \cdot \exp \left( -\frac{(x_i - \mu_y)^2}{2 \sigma_y^2} \right)$$

## Predict Technique

$$y = argmax_y log(P(x_1|y)) + loglog(P(x_2|y)) + ... + log(P(x_n|y)) + log(P(y)) $$

## Gradient Descent

## Cost Function


## Priors / Initial Probability
$$ Priors =  P(A) = P(y = k) = \frac{\text{jumlah data dengan kelas } k}{\text{total data}} $$
Untuk menghindari perkalian yang menghasilkan bilangan kecil dan underflow numerik, maka kita melalukan operasi log untuk priors $log(P(y))$