# Logistic Regression

## Approximation
$$\hat{y} = h_\theta(x) = \frac{1}{1 + e^{-(wx + b)}}$$

$$y(x) = \frac{1}{1 + e^{-x}}$$

## Gradient Descent
$$dw = \frac{1}{n} X^\top \left(\hat{y} - y\right)$$

$$db = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{y}^{(i)} - y^{(i)}\right)$$

## Cost Function
Cross Entropy:

$$
J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

## Update Rules
$$w = w - \alpha * dw$$

$$b = b - \alpha * db$$