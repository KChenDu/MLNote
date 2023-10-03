1. 概率分布
    1. 离散型变量和概率质量函数
        - 均匀分布：$P(x = x_i) = \frac1k$
        $$\sum_iP(x = x_i) = \sum_i\frac1k = \frac kk = 1$$
    2. 边缘概率
        - 求和法则：
            $$\forall x\in X, P(X = x) = \sum_y P(X = x, Y = y)$$
            $$p(x) = \int p(x, y)\d y$$