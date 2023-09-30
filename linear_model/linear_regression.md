1. 线性回归
    $$f(\mathbf x_i) = \mathbf \omega^T\mathbf x_i + b\Rightarrow f(\mathbf X) =\mathbf X\hat{\mathbf \omega}$$
    $$\hat{\mathbf \omega}\ast = \argmin_{\hat{\mathbf \omega}}(\mathbf y - \mathbf X\hat{\mathbf \omega})^T(\mathbf y - \mathbf X\hat{\mathbf \omega})$$
    $$\frac{\partial E_{\hat{\mathbf \omega}}}{\partial \hat{\mathbf \omega}} = 2\mathbf X^T(\mathbf X\hat{\mathbf \omega} - \mathbf y)\Rightarrow \hat{\mathbf \omega}\ast = (\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y$$
2. [实战](linear_regression.ipynb)