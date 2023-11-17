1. 随机变量：是可以随机地取不同值的变量。随机变量可以是离散的或者连续的。离散随机变量拥有有限或者可数无限多的状态。连续随机变量伴随着实数值
2. 概率分布
    1. 离散型变量和概率质量函数

        概率质量函数可以同时作用于多个随机变量。这种多个变量的概率分布被称为**联合概率分布**。
        
        如果一个函数$P$是随机变量$X$的PMF，必须满足下面这几个条件：
        - $P$的定义域必须是$X$所有可能状态的集合。
        - $\forall x \in X, 0 \le P(x) \le 1$。不可能发生的事件概率为0，并且不存在比这概率更低的状态。类似的，能够确保一定发生的事件概率为1，而且不存在比这概率更高的状态
        - $\sum_{x \in X}P(x) = 1$。我们把这条性质称之为**归一化的**。如果没有这条性质，当我们计算很多事件其中之一发生的概率时可能会得到大于1的概率
        - 均匀分布：$P(X = x_i) = \frac1k \Rightarrow \sum_iP(X = x_i) = \sum_i\frac1k = \frac kk = 1$
    2. 连续型变量和概率密度函数
	    如果一个函数$p$是概率密度函数，必须满足下面这几个条件：
        - $p$的定义域必须是$X$所有可能状态的集合
        - $\forall x \in X, p(x) \ge 0$。注意，我们并不要求$p(x) \le 1$
        - $\int p(x)dx = 1$
    5. 边缘概率
        - 求和法则：
            $$\forall x \in X, P(X = x) = \sum_y P(X = x, Y = y)$$
            $$p(x) = \int p(x, y)\mathrm dy$$
3. 条件概率：$P(Y = y | X = x) = \frac{P(Y = y, X = x)}{P(X = x)}$
4. 条件概率的链式法则：$P(X^{(1)}, ..., X^{(n)}) = P(X^{(1)})\prod_{i = 2}^nP(X^{(i)} | X^{(1)}, ..., X^{(i - 1)})$
5. 独立性和条件独立性
    - 相互独立的：$\forall x \in X, y \in Y, p(X = x, Y = y) = p(X = x)p(Y = y)$
    - 条件独立的：$\forall x \in X, y \in Y, z \in Z, p(X = x, Y = y, Z = z) = p(X = x | Z = z)p(Y = y | Z = z)$
6. 期望、方差和协方差
    - 期望：
        $$\mathbb E_{X \sim P}[f(x)] = \sum_x P(x)f(x)$$
        $$\mathbb E_{x \sim p}[f(x)] = \int p(x)f(x)\mathrm dx$$
        $$\mathbb E_X[\alpha f(x) + \beta g(x)] = \alpha\mathbb E_X[f(x)] + \beta\mathbb E_X[g(x)]$$
    - 方差：$\mathrm{Var}(f(x)) = \mathbb E[(f(x) - \mathbb E[f(x)])^2]$
    - 方差的平方根被称为**标准差**
    - 协方差：$\mathrm{Cov}(f(x), g(y)) = \mathbb E[(f(x) - \mathbb E[f(x)])(g(y) - \mathbb E[g(y)])]$
    - 协方差矩阵：
        $$\mathrm{Cov}(\mathbf X)_{i, j} = \mathrm{Cov}(X_i, X_j)$$
        $$\mathrm{Cov}(X_i, X_i) = \mathrm{Var}(X_i)$$
7. 常用概率分布
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
        当我们由于缺乏关于某个实数上分布的先验知识而不知道该选择怎样的形式时，正态分布是默认的比较好的选择，其中有两个原因。
        - 第一，我们想要建模的很多分布的真实情况是比较接近正态分布的。中心极限定理说明很多独立随机变量的和近似服从正态分布。这意味着在实际中，很多复杂系统都可以被成功地建模成正态分布的噪声，即使系统可以被分解成一些更结构化的部分。
        - 第二，在具有相同方差的所有可能的概率分布中，正态分布在实数上具有最大的不确定性。因此，我们可以认为正态分布是对模型加入的先验知识量最少的分布。
        - 多维正态分布：$\mathcal N(\mathbf x; \mathbf\mu, \mathbf\Sigma) = \sqrt{\frac1{(2\pi)^n\det \mathbf\Sigma}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\Sigma^{-1}(\mathbf x - \mathbf\mu)) \Rightarrow \mathcal N(\mathbf x; \mathbf\mu, \mathbf\beta^{-1}) = \sqrt{\frac{\det\mathbf\beta}{(2\pi)^n}}\exp(-\frac12(\mathbf x - \mathbf\mu)^\top\mathbf\beta(\mathbf x - \mathbf\mu))$
    4. 指数分布和Laplace分布
        - 指数分布：$p(x; \lambda) = \lambda\mathbf 1_{x \ge 0}\exp(-\lambda x)$
        - Laplace分布：$\mathrm{Laplace}(x; \mu, \gamma) = \frac1{2\gamma}\exp(-\frac{|x - \mu|}\gamma)$
    5. Dirac分布和经验分布
    6. 分布的混合
        $$P(X) = \sum_iP(c = i)P(x | c = i)$$
        混合模型使我们能够一瞥以后会用到的一个非常重要的概念——**潜变量**。
        
        一个非常强大且常见的混合模型是高斯混合模型，它的组件$p(\mathbf X | c = i)$是高斯分布。每个组件都有各自的参数，均值$\mathbf\mu^{(i)}$和协方差矩阵$\Sigma^{(i)}$。除了均值和协方差以外， 高斯混合模型的参数指明了给每个组件$i$的**先验概率**$\alpha_i = P(c = i)$。“先验”一词表明了在观测到$X$之前传递给模型关于$c$的信念。作为对比，$P(c | x) 是**后验概率**，因为它是在观测到$X$之后进行计算的。高斯混合模型是概率密度的**万能近似器**，在这种意义下，任何平滑的概率密度都可以用具有足够多组件的高斯混合模型以任意精度来逼近
8. 常用函数的有用性质
    - **logistic sigmoid**函数：
        $$\sigma(x) = \frac1{1 + \exp(-x)} \Rightarrow
        \begin{cases}
            \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(0)} \\
            \frac d{dx}\sigma(x) = \sigma(x)(1 - \sigma(x)) \\
            1 - \sigma(x) = \sigma(-x) \\
            \forall x \in (0, 1), \sigma^{-1}(x) = \log(\exp(x) - 1)
        \end{cases}$$
        sigmoid函数在变量取绝对值非常大的正值或负值时会出现饱和现象，意味着函数会变得很平，并且对输入的微小改变会变得不敏感

9. 贝叶斯规则：$P(X | Y) = \frac{P(X)P(Y | X)}{P(Y)}$
10. 连续型变量的技术细节
11. 信息论

    我们想要通过基本想法来量化信息。特别地，
    - 非常可能发生的事件信息量要比较少，并且极端情况下，确保能够发生的事件应该没有信息量
    - 较不可能发生的事件具有更高的信息量
    - 独立事件应具有增量的信息。例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次硬币正面朝上的信息量的两倍
    
    自信息：$I(x) = -\log P(x)$
    
    我们可以用**香农熵**来对整个概率分布中的不确定性总量进行量化：$H(x) = \mathbb E_{x \sim P}(I(x)) = -\mathbb E_{x \sim P}(\log P(x))$
	如果我们对于同一个随机变量$x$有两个单独的概率分布$P(x)$和$Q(x)$，我们可以使用KL散度来衡量这两个分布的差异：$D_{KL}(P\|Q) = \mathbb E_{x \sim P}[\log\frac{P(x)}{Q(x)}] = \mathbb E_{x \sim P}[\log P(x) - \log Q(x)]$
    
    在离散型变量的情况下，KL散度衡量的是，当我们使用一种被设计成能够使得概率分布$Q$产生的消息的长度最小的编码，发送包含由概率分布$P$产生的符号的消息时，所需要的额外信息量
	交叉熵：$H(P, Q) = -\mathbb E_{x \sim P}\log Q(x)$

[返回](readme.md)