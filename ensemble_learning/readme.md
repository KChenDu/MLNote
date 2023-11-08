1. 个体与集成

    - 一般结构：先产生一组“个体学习器”，再用某种策略将它们结合起来。个体学习器通常由一个现有的学习算法从训练数据产生

    同质集成中的个体学习器亦称“基学习器”，相应的学习算法称为“基学习算法”。集成也可包含不同类型的个体学习器，例如同时包含决策树和神经网络，这样的集成是“异质”的。异质集成中的个体学习器由不同的学习算法生成，这时就不再有基学习算法

    要获得好的集成，个体学习器应“好而不同”，即个体学习器要有一定的“准确性”，即学习器不能太坏，并且要有“多样性”，即学习器间具有差异

    $$
    \begin{cases}
        P(h_i(x) \neq f(x)) = \epsilon \\
        H(x) = \mathrm{sign}(\sum_{i = 1}^Th_i(x))
    \end{cases}
    \Rightarrow P(H(x) \neq f(x)) = \sum_{k = 0}^{\lfloor T / 2\rfloor}{T\choose k}(1 - \epsilon)^k\epsilon^{T - k} \leq \exp(-\frac12T(1 - 2\epsilon)^2)$$
2. [Boosting](gradient_boosting.ipynb)

    先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值$T$，最终将这$T$个基学习器进行加权结合
    - [AdaBoost](adaboost.ipynb)
3. Bagging与随机森林
    1. [Bagging](bagging.ipynb)
    2. [随机森林](random_forest.ipynb)
4. 结合策略
    1. 平均法
        - 简单平均法：$H(\mathbf x) = \frac1T\sum_{i = 1}^Th_i(\mathbf x)$
        - 加权平均法：$H(\mathbf x) = \sum_{i = 1}^Tw_ih_i(\mathbf x)$
    2. [投票法](voting_classifiers.ipynb)
    3. [学习法](stacking.ipynb)

[返回](../readme.md)