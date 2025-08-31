# Linear Regression

$$f(w, b) = wx + b$$

## Gradient Descent
$$dw = \frac{1}{n} X^\top \left(\hat{y} - y\right)$$

$$db = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{y}^{(i)} - y^{(i)}\right)$$

## Cost Function

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

## Update Rules
$$w = w - \alpha * dw$$

$$b = b - \alpha * db$$