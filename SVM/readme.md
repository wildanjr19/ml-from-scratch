# Support Vector Machine

## Linear Model
$$w \cdot x - b = 0 $$

$$w \cdot x_i - b \geq 1 \quad \text {if }  y_i = 1$$

$$w \cdot x_i - b \leq -1 \quad \text {if }  y_i = -1$$

$$y_i(w \cdot x_i - b) \geq 1$$

## Cost Function
Hinge Loss
$$l = max(0, 1 - y_i(w \cdot x_i - b))$$

$$
\ell =
\begin{cases}
    0 & \text{if } y \cdot f(x) \geq 1 \\
    1 - y \cdot f(x) & \text{otherwise.}
\end{cases}
$$  

### Add Regularization
$$
J = \lambda||w||^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 -  y_i(w \cdot x_i - b))
$$  

- if $y_i \cdot f(x) \geq 1$:
$$J_i = \lambda||w||^2$$

- else:
$$J_i = \lambda||w||^2 + 1 - y_i(w \cdot x_i - b)$$

## Gradients
- if $y_i \cdot f(x) \geq 1$:
$$
\frac{dJ_i}{dw_k} = 2\lambda w_k
$$
$$
\frac{dJ_i}{db} = 0
$$

- else:
$$
\frac{dJ_i}{dw_k} = 2\lambda w_k -  y_i \cdot x_i
$$
$$
\frac{dJ_i}{db} = y_i
$$

## Update Rule
Untuk setiap sampel $x_i$:  
$w = w - \alpha \cdot dw$

$b = b - \alpha \cdot db$

## Code Notes
`np.sign` mengembalikan tanda positif atau negatif (1, -1)