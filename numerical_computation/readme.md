1. 上溢和下溢
2. 病态条件
3. 基于梯度的优化方法
    
    导数对于最小化一个函数很有用，因为它告诉我们如何更改$x$来略微地改善$y$。例如，我们知道对于足够小的$\epsilon$来说，$f(x − \epsilon\mathrm{sign}(f^\prime(x)))$是比$f(x)$小的。因此我们可以将$x$往导数的反方向移动一小步来减小$f(x)$。这种技术被称为**梯度下降**
    - $f^\prime(x) = 0$的点称为**临界点**
    - 一个**局部极小点**意味着这个点的$f(x)$小于所有邻近点，因此不可能通过移动无穷小的步长来减小$f(x)$。
    - 一个**局部极大点**意味着这个点的$f(x)$大于所有邻近点，因此不可能通过移动无穷小的步长来增大$f(x)$
    - 有些临界点既不是最小点也不是最大点。这些点被称为**鞍点**
    - 使$f(x)$取得绝对的最小值（相对所有其他值）的点是**全局最小点**
    - **偏导数**$\frac\partial{\partial x_i} f(\mathbf x)$衡量点$\mathbf x$处只有$x_i$增加时$f(\mathbf x)$如何变化。
    - **梯度**是相对一个向量求导的导数：$f$的导数是包含所有偏导数的向量，记为$\nabla_{\mathbf x}f(\mathbf x)$
    - 我们在负梯度方向上移动可以减小$f$。这被称为**最速下降法**或**梯度下降**
        $$\mathbf x^\prime = \mathbf x - \epsilon\nabla_{\mathbf x}f(\mathbf x)$$
        其中$\epsilon$为**学习率**
4. 梯度之上：Jacobian和Hessian矩阵
    
    如果我们有一个函数：$f: \mathbb R^m \rightarrow \mathbb R^n$，$f$的Jacobian矩阵$J \in \mathbb R^{n \times m}$定义为$J_{i, j} = \frac\partial{\partial x_j}f(\mathbf x)_i$

    Hessian矩阵$H(f)(\mathbf x)$定义为
    $$H(f)(\mathbf x)_{i, j} = \frac{\partial^2}{\partial x_j\partial x_i}f(\mathbf x)$$

    我们可以通过（方向）二阶导数预期一个梯度下降步骤能表现得多好。我们在当前点$x^{(0)}$处作函数$f(x)$的近似二阶泰勒级数：
    $$f(\mathbf x) \approx f(\mathbf x^{(0)}) + (\mathbf x - \mathbf x^{(0)})^\top\mathbf g + \frac12(\mathbf x - \mathbf x^{(0)})^\top H(\mathbf x - \mathbf x^{(0)}) \Rightarrow f(\mathbf x^{(0)} - \epsilon \mathbf g) \approx f(\mathbf x^{(0)}) - \epsilon \mathbf g^\top\mathbf g + \frac12\epsilon^2\mathbf g^\top H\mathbf g$$
    - 当$\mathbf g^\top H\mathbf g$为正时，通过计算可得，使近似泰勒级数下降最多的最优步长为$\epsilon^* =\frac{\mathbf g^\top\mathbf g}{\mathbf g^\top H\mathbf g}$
    - 牛顿法：$f(\mathbf x) \approx f(\mathbf x^{(0)}) + (\mathbf x - \mathbf x^{(0)})^\top\nabla_{\mathbf x}f(\mathbf x^{(0)}) + \frac12(\mathbf x - \mathbf x^{(0)})^\top H(f)(\mathbf x^{(0)})(\mathbf x - \mathbf x^{(0)})$
    
    在深度学习的背景下，限制函数满足Lipschitz连续或其导数Lipschitz连续可以获得一些保证。Lipschitz连续函数的变化速度以Lipschitz常数$\mathcal L$为界：$\forall\mathbf x, \forall\mathbf y, |f(\mathbf x) - f(\mathbf y) \le \mathcal L\|\mathbf x - \mathbf y\|_2$

    最成功的特定优化领域或许是**凸优化**。凸优化通过更强的限制提供更多的保证
5. 约束优化

    广义Lagrangian可以如下定义：$L(\mathbf x, \mathbf\lambda ,\mathbf\alpha) = f(\mathbf x) + \sum_i\lambda_ig^{(i)}(\mathbf x) + \sum_j\alpha_jh^{(j)}(\mathbf x)$

    要解决约束最大化问题，我们可以构造$−f(x)$的广义Lagrange函数，从而导致以下优化问题：$\min_{\mathbf x}\max_{\mathbf\lambda}\max_{\mathbf\alpha, \mathbf\alpha \ge 0}-f(\mathbf x) + \sum_i\lambda_ig^{(i)}(\mathbf x) + \sum_j\alpha_jh^{(j)}(\mathbf x)$

    我们也可将其转换为在外层最大化的问题：$\max_{\mathbf x}\min_{\mathbf\lambda}\min_{\mathbf\alpha, \mathbf\alpha \ge 0}f(\mathbf x) + \sum_i\lambda_ig^{(i)}(\mathbf x) - \sum_j\alpha_jh^{(j)}(\mathbf x)$

    我们可以使用一组简单的性质来描述约束优化问题的最优点。这些性质称为**Karush–Kuhn–Tucker**（KKT）条件。这些是确定一个点是最优点的必要条件，但不一定是充分条件。这些条件是：
    - 广义Lagrangian的梯度为零
    - 所有关于$\mathbf x$和KKT乘子的约束都满足
    - 不等式约束显示的“互补松弛性”：$\mathbf\alpha \odot h(\mathbf x) = 0$。

[返回](../readme.md)