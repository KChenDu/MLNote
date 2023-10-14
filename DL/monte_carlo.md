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
    $$\hat s_p = \frac1n\sum_{i = 1, x^{(i)} \sim p}^nf(x^{(i)}) \Rightarrow \hat s_q = \frac1n\sum_{i = 1, x^{(i)} \sim q}^n\frac{p(x^{(i)})f(x^{(i)})}{q(x^{(i)})}$$
    $$\mathbb E_q[\hat s_q] = \mathbb E_p[\hat s_p] = s$$
    $$\mathrm{Var}[\hat s_q] = \mathrm{Var}[\frac{p(\mathbf x)f(\mathbf x)}{q(\mathbf x)}] / n$$
    - 方差想要取到最小值，$q$需要满足$q^\ast(x) = \frac{p(x)|f(x)|}Z$。在这里$Z$表示归一化常数。选择适当的$Z$使得$q^\ast(x)$之和或者积分为1
    - 有偏重要采样：$\hat s_{\mathrm{BIS}} = \frac{\sum_{i = 1}^n\frac{p(x^{(i)})}{q(x^{(i)})}f(x^{(i)})}{\sum_{i = 1}^n\frac{p(x^{(i)})}{q(x^{(i)})}} = \frac{\sum_{i = 1}^n\frac{p(x^{(i)})}{\tilde q(x^{(i)})}f(x^{(i)})}{\sum_{i = 1}^n\frac{p(x^{(i)})}{\tilde q(x^{(i)})}} = \frac{\sum_{i = 1}^n\frac{\tilde p(x^{(i)})}{\tilde q(x^{(i)})}f(x^{(i)})}{\sum_{i = 1}^n\frac{\tilde p(x^{(i)})}{\tilde q(x^{(i)})}}$，其中$\tilde p$和$\tilde q$分别是$p$和$q$的未经归一化的形式，$x^{(i)}$是从分布$q$中抽取的样本

[返回](readme.md)