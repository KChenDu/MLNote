3. EM算法
    $$LL(\Theta | X, Z) = \ln P(X, Z | \Theta)$$
    对$Z$计算期望，来最大化已观测数据的对数“边际似然”$LL(\Theta) = \ln\sum_ZP(X, Z | \Theta)$
    
    以初始值$\Theta^0$为起点，可迭代执行以下步骤直至收敛：

    - 基于$\Theta$推断隐变量$Z$的期望，记为$Z^t$
    - 基于已观测变量$X$和$Z$对参数$\Theta$做极大似然估计，记为$\Theta^t$
    
    若我们不是取$Z$的期望，而是基于$\Theta^t$计算变量$Z$的概率分布$P(Z | X, \Theta^t)$，则EM算法的两个步骤是：
    - E步：以当前参数$\Theta^t$推断隐变量分布$P(Z | X, \Theta^t)$，并计算对数似然$LL(\Theta | X, Z)$关于$Z$的期望$Q(\Theta | \Theta^t) = \mathbb E_{Z | X, \Theta^t}LL(\Theta | X, Z)$
    - M步：寻找参数最大化期望似然，即$\Theta^{t + 1} = \argmax_\Theta Q(\Theta | \Theta^t)$

[返回](readme.md)