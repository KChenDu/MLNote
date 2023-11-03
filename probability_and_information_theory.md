1. 概率分布
    1. 离散型变量和概率质量函数
        - 均匀分布：$P(X = x_i) = \frac1k \Rightarrow \sum_iP(X = x_i) = \sum_i\frac1k = \frac kk = 1$
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
    - 方差：$\mathrm{Var}(f(x)) = E[(f(x) - E[f(x)])^2]$
    - 协方差：$\mathrm{Cov}(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]$
    - 协方差矩阵：
        $$\mathrm{Cov}(x)_{i, j} = \mathrm{Cov}(x_i, x_j)$$
        $$\mathrm{Cov}(x_i, x_i) = \mathrm{Var}(x_i)$$
6. 常用概率分布
    1. Bernoulli分布
        $$\begin{cases}
            P(X = 1) = \phi \\
            P(X = 0) = 1 - \phi
        \end{cases}
        \Rightarrow
        \begin{cases}
            P(X = x) = \phi^x(1 - \phi)^{1 - x} \\
            E[X] = \phi \\
            \mathrm{Var}(X) = \phi(1 - \phi)
        \end{cases}$$
    2. Multinoulli分布：在具有$k$个不同状态的单个离散型随机变量上的分布，其中$k$是一个有限值
    3. 高斯分布：
        $$\mathcal N(x; \mu, \sigma^2) = \sqrt{\frac1{2\pi\sigma^2}}\exp(-\frac1{2\sigma^2}(x - \mu)^2) \Rightarrow \mathcal N(x; \mu, \beta^{-1}) = \sqrt{\frac\beta{2\pi}}\exp(-\frac12\beta(x - \mu)^2)$$
        - 多维正态分布：$\mathcal N(\mathbf x; \mathbf\mu, \mathbf\Sigma) = \sqrt{\frac1{(2\pi)^n\det \mathbf\Sigma}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\Sigma^{-1}(\mathbf x - \mathbf\mu)) \Rightarrow \mathcal N(\mathbf x; \mathbf\mu, \mathbf\beta^{-1}) = \sqrt{\frac{\det\mathbf\beta}{(2\pi)^n}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\beta(\mathbf x - \mathbf\mu))$
    6. 分布的混合
        $$P(x) = P(c = i)P(x | c = i)$$
        混合模型使我们能够一瞥以后会用到的一个非常重要的概念——潜变量。一个非常强大且常见的混合模型是高斯混合模型，它的组件$p(x | c = i)$是高斯分布
7. 常用函数的有用性质
    - **logistic sigmoid**函数：$\sigma(x) = \frac1{1 + \exp(-x)}$
        
        sigmoid函数在变量取绝对值非常大的正值或负值时会出现饱和现象，意味着函数会变得很平，并且对输入的微小改变会变得不敏感
8. 贝叶斯规则：$P(X | Y) = \frac{P(X)P(Y | X)}{P(Y)}$
9. 连续型变量的技术细节
10. 信息论

    我们想要通过基本想法来量化信息。特别地，
    - 非常可能发生的事件信息量要比较少，并且极端情况下，确保能够发生的事件应该没有信息量
    - 较不可能发生的事件具有更高的信息量
    - 独立事件应具有增量的信息。例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次硬币正面朝上的信息量的两倍
    
    自信息：$I(x) = -\log P(x)$
    
    我们可以用**香农熵**来对整个概率分布中的不确定性总量进行量化：
    $$H(x) = \mathbb E_{x \sim P}(I(x)) = -\mathbb E_{x \sim P}(\log P(x))$$

    

[返回](readme.md)