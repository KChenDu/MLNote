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
    $$$$

[返回](../readme.md)