1. 朴素贝叶斯分类器
    $$P(c | \mathbf x) = \frac{P(c)P(\mathbf x | c)}{P(\mathbf x)} = \frac{P(c)}{P(\mathbf x)}\prod_{i = 1}^dP(x_i | c) \Rightarrow h_{nb}(\mathbf x) = \argmax_{c \in \mathcal Y}P(c)\prod_{i = 1}^dP(x_i | c)$$
    ```mermaid
    graph
    y((y))

    y --> x1((x1))
    y --> x2((x2))
    y --> x3((x3))
    y --> xd((xd))
    ```
    令$D_c$表示训练集$D$中第$c$类样本组成的集合，若有充足的独立同分布样本，则可容易地估计出类先验概率$P(x_i | c)$可估计为$P(c) = \frac{|D_c|}{|D|}$
    
    对离散属性而言，令$D_{c, x_i}$表示$D_c$中在第$i$个属性上取值为$x_i$的样本组成的集合，则条件概率$P(x_i | c) = \frac{|D_{c, x_i}|}{|D_c|}$
    
    对连续属性可考虑概率密度函数，假定$p(x_i | c) \sim \mathcal N(\mu_{c, i}, \sigma_{c, i}^2)$其中$\mu_{c, i}$和$\sigma_{c, i}^2$分别是第$c$类样本在第$i$个属性上取值的均值和方差，则有$p(x_i | c) = \frac1{\sqrt{2\pi}\sigma_{c, i}}\exp(-\frac{(x_i - \mu_{c, i})^2}{2\sigma_{c, i}^2})$
    
    为了避免其他属性携带的信息被训练集中未出现的属性值“抹去”，在估计概率值时通常要进行“平滑”，常用“拉普拉斯修正”。具体来说，令$N$表示训练集$D$中可能的类别数，$N_i$表示第$i$个属性可能的取值数，则
    $$\hat P(c) = \frac{|D_c|  +1}{|D| + N}$$
    $$\hat P(x_i | c) = \frac{|D_{c, x_i}|  1}{|D_c| + N_i}$$
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