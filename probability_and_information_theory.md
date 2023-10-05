1. 概率分布
    1. 离散型变量和概率质量函数
        - 均匀分布：$P(X = x_i) = \frac1k$
        $$\sum_iP(X = x_i) = \sum_i\frac1k = \frac kk = 1$$
    2. 边缘概率
        - 求和法则：
            $$\forall x \in X, P(X = x) = \sum_y P(X = x, Y = y)$$
            $$p(x) = \int p(x, y)\mathrm dy$$
2. 条件概率
    $$P(Y = y | X = x) = \frac{P(Y = y, X = x)}{P(X = x)}$$
3. 条件概率的链式法则
    $$P(x^{(1)}, ..., x^{(n)}) = P(x^{(1)})\prod_{i = 2}^nP(x^{(i)} | x^{(1)}, ..., x^{(i - 1)})$$
4. 独立性和条件独立性
    $$\forall x \in X, y \in Y, p(X = x, Y = y) = p(X = x)p(Y = y)$$
    $$\forall x \in X, y \in Y, z \in Z, p(X = x, Y = y, Z = z) = p(X = x | Z = z)p(Y = y | Z = z)$$
5. 期望、方差和协方差
    - 期望：
        $$E[f(x)] = \sum_x P(x)f(x)$$
        $$E[f(x)] = \int p(x)f(x)\mathrm dx$$
        $$E[\alpha f(x) + \beta g(x)] = \alpha E[f(x)] + \beta E[g(x)]$$
    - 方差：$Var(f(x)) = E[(f(x) - E[f(x)])^2]$
    - 协方差：$Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]$
    - 协方差矩阵：
        $$Cov(x)_{i, j} = Cov(x_i, x_j)$$
        $$Cov(x_i, x_i) = Var(x_i)$$
6. 常用概率分布
    1. Bernoulli分布
        $$P(X = 1) = \phi$$
        $$P(X = 0) = 1 - \phi$$
        $$P(X = x) = \phi^x(1 - \phi)^{1 - x}$$
        $$E[X] = \phi$$
        $$Var(X) = \phi(1 - \phi)$$
    2. Multinoulli分布：？？？
    3. 高斯分布：
        $$\mathcal N(x; \mu, \sigma^2) = \sqrt{\frac1{2\pi\sigma^2}}\exp(-\frac1{2\sigma^2}(x - \mu)^2) \Rightarrow \mathcal N(x; \mu, \beta^{-1}) = \sqrt{\frac\beta{2\pi}}\exp(-\frac12\beta(x - \mu)^2)$$
        - 多维正态分布：$\mathcal N(\mathbf x; \mathbf\mu, \mathbf\Sigma) = \sqrt{\frac1{(2\pi)^n\det \mathbf\Sigma}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\Sigma^{-1}(\mathbf x - \mathbf\mu)) \Rightarrow \mathcal N(\mathbf x; \mathbf\mu, \mathbf\beta^{-1}) = \sqrt{\frac{\det\mathbf\beta}{(2\pi)^n}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\beta(\mathbf x - \mathbf\mu))$
7. 常用函数的有用性质
8. 贝叶斯规则：$P(X | Y) = \frac{P(X)P(Y | X)}{P(Y)}$

[返回](readme.md)