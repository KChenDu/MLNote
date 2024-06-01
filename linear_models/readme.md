1. 基本形式：$f(\mathbf x) = \mathbf w^\top\mathbf x + b$
2. [线性回归](linear_regression.ipynb)
3. [对数几率回归](logistic_regression.ipynb)
4. 线性判别分析
    
    给定数据集$D = \{(\mathbf x_i, y_i)\}_{i = 1}^m, y_i \in \{0， 1\}$，令$\mathbf X_i$、$\mathbf\mu_i$、$\Sigma_i$分别表示第$i \in \{0，1\}$类示例的集合、均值向量、协方差矩阵。若将数据投影到直线$\mathbf w$上，则两类样本的中心在直线上的投影分别为$\mathbf w^\top \mu_0$和$\mathbf w^\top \mu_1$；若将所有样本点都投影到直线上，则两类样本的协方差分别为$\mathbf w^\top \Sigma_0\mathbf w$和$\mathbf w^\top \Sigma_1\mathbf w$。由于直线是一维空间，因此$\mathbf w^\top \mu_0$、$\mathbf w^\top \mu_1$、$\mathbf w^\top \Sigma_0\mathbf w$和$\mathbf w^\top \Sigma_1\mathbf w$均为实数
    - 类内散度矩阵：$S_w = \Sigma_0 + \Sigma_1 = \sum_{\mathbf x \in X_0}(\mathbf x - \mathbf\mu_0)(\mathbf x - \mathbf\mu_0)^\top + \sum_{\mathbf x \in X_1}(\mathbf x - \mathbf\mu_1)(\mathbf x - \mathbf\mu_1)^\top$
    - 类问散度矩阵：$S_b = (\mathbf\mu_0 - \mathbf\mu_1)(\mathbf\mu_0 - \mathbf\mu_1)^\top$
    - 欲最大化的目标：$J = \frac{\|\mathbf w^\top \mu_0 - \mathbf w^\top \mu_1\|_2^2}{\mathbf w^\top \Sigma_0\mathbf w + \mathbf w^\top \Sigma_1\mathbf w} = \frac{\mathbf w^\top (\mu_0 - \mu_1)(\mu_0 - \mu_1)^\top\mathbf w}{\mathbf w^\top(\Sigma_0 + \Sigma_1)\mathbf w} = \frac{\mathbf w^\top S_b\mathbf w}{\mathbf w^\top S_w\mathbf w} \Rightarrow \mathbf w = S_w^{-1}(\mathbf\mu_0 - \mathbf\mu_1)$

    可以将LDA推广到多分类任务中。假定存在$N$个类，且第$i$类示例数为$m_i$。我们先定义“全局散度矩阵”$S_t = S_b + S_w = \sum_{i = 1}^m(\mathbf x_i - \mathbf\mu)(\mathbf x_i - \mathbf\mu)^\top$，其中$\mathbf\mu$是所有示例的均值向量。将类内散度矩阵$S_w$重定义为每个类别的散度矩阵之和，即$S_w = \Sigma_{i = 1}^NS_{w_i}$，其中$S_{w_i} = \sum_{\mathbf x \in X_i}(\mathbf x - \mathbf\mu_i)(\mathbf x - \mathbf\mu_i)^\top \Rightarrow S_b = S_t - S_w = \sum_{i = 1}^Nm_i(\mathbf\mu_i - \mathbf\mu)(\mathbf\mu_i - \mathbf\mu)^\top$

    显然，多分类LDA可以有多种实现方法：使用$S_b$，$S_w$，$S_t$三者中的任何两个即可。常见的一种实现是采用优化目标$\max_{W}\frac{W^\top S_bW}{W^\top S_wW}$
    
5. [Softmax回归](softmax_regression.ipynb)

[返回](../readme.md)