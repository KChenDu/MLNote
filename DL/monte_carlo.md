1. 采样和蒙特卡罗方法
    1. 为什么需要采样
        - 以较小的代价近似许多项的和或某个积分
        - 加速一些很费时却易于处理的求和估计、
        - 近似一个难以处理的求和或积分
    2. 蒙特卡罗采样的基础
        $$
        \begin{cases}
            s = \sum_x p(x)f(x) = E_p[f(\mathbf x)] \\
            s = \int p(x)f(x)\mathrm dx = E_p[f(\mathbf x)]
        \end{cases}
        \Rightarrow \hat s_n = \frac1n\sum_{i = 1}^nf(x^{(i)})$$
        下面几个性质表明了这种近似的合理性：
        $$\mathbb E[\hat s_n] = \frac 1n\sum_{i = 1}^n\mathbb E[f(x^{(i)})] = \frac 1n\sum_{i = 1}^ns = s$$
        $$\lim_{n \rightarrow \infty}\hat s_n = s$$
        $$\mathrm{Var}[\hat s_n] = \frac1{n^2}\sum_{i = 1}^n\mathrm{Var}[f(\mathbf x)] = \frac{\mathrm{Var}[f(\mathbf x)]}n$$
2. 重要采样


[返回](readme.md)