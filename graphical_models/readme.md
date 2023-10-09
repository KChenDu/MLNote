1. 隐马尔可夫模型
    $$P(x_1, y_1, ..., x_n, y_n) = P(y_1)P(x_1 | y1)\prod_{i = 2}^nP(yi | y_{i - 1})P(x_i | y_i)$$
    - 状态转移概率：$a_{ij} = P(y_{t + 1} = s_j | y_t = s_i), 1 \leq i, j \leq N$
    - 输出观测概率：$b_{ij} = P(x_t = o_j | y_t = s_i), 1 \leq i \leq N, 1 \leq i \leq N, i \leq j \leq M$
    - 初始状态概率：$\pi_i = P(y_1 = s_i)$
2. 马尔可夫随机场