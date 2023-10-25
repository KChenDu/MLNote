3. 基本算法
    1. [随机梯度下降](sgd.ipynb)
        - 保证SGD收敛的一个充分条件是$\sum_{k = 1}^\infty\epsilon_k = \infty$且$\sum_{k = 1}^\infty\epsilon_k^2 < \infty$
        
            实践中，一般会线性衰减学习率直到第$\tau$次迭代：
            $$\epsilon_k = (1 - \alpha)\epsilon_0 + \alpha\epsilon_\tau$$
    2. 动量

        从形式上看，动量算法引入了变量$\mathbf v$充当速度角色——它代表参数在参数空间移动的方向和速率。速度被设为负梯度的指数衰减平均。更新规则如下：
        $$\mathbf v \leftarrow \alpha\mathbf v - \epsilon\nabla_{\mathbf\theta}(\frac1m\sum_{i = 1}^mL(f(\mathbf x^{(i)}; \mathbf\theta), \mathbf y^{(i)})) \\
        \mathbf\theta \leftarrow \mathbf\theta + \mathbf v$$
        - 步长大小：$\frac{\epsilon\|\mathbf g\|}{1 - \alpha}$
5. 自适应学习率算法
    1. AdaGrad

[返回](readme.md)