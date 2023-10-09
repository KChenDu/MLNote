1. 评估方法
    1. 留出法：直接将数据集$D$划分为两个互斥的集合，其中一个集合作为训练集$S$，另一个作为测试集$T$，即$D = S \cup T$，$S \cap T = \empty$。在$S$上训练出模型后，用$T$来评估其测试误差，作为对泛化误差的估计。
        - 训练/测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程引入额外的偏差而对最终结果产生影响
        - 即使在给定训练/测试集的样本比例后，仍存在多种划分方式对初始数据集$D$进行分割
    2. 交叉验证法：先将数据集$D$划分为$k$个大小相似的互斥子集，即$D = D_1 \cup D_2 \cup ... \cup D_k$。每个子集$D_i$都尽可能保持数据分布的一致性，即从$D$中通过分层采样得到。然后，每次用$k - 1$个子集的并集作为训练集，余下的那个子集作为测试集；这样就可获得$k$组训练/测试集，从而可进行$k$次训练和测试，最终返回的是这$k$个测试结果的均值。
        - 与留出法相似，将数据集$D$划分为$k$个子集同样存在多种划分方式
    3. 自助法：給定包含m个样本的数据集$D$，我们对它进行采样产生数据集$D'$：每次随机从$D$中挑选一个样本，将其拷贝放入$D'$，然后再将该样本放回初始数据集$D$中，使得该样本在下次采样时仍有可能被采到；这个过程重复执行$m$次后，我们就得到了包含$m$个样本的数据集$D'$，这就是自助采样的结果。
        - 样本在$m$次采样中始终不被采到的概率是$(1 - \frac1m)^m$，取极限得到$\lim_{m \rightarrow \infty}(1 - \frac1m)^m = \frac1e$
2. 性能度量
    - 均方误差
        $$E(f; D) = \frac1m\sum_{i = 1}^m(f(\mathbf x_i) - y_i)^2$$
        $$E(f; D) = \int_{x \sim D}(f(\mathbf x) - y)^2p(\mathbf x)\mathrm d\mathbf x$$
    1. 错误率与精度
        - 错误率
            $$E(f; D) = \frac1m\sum_{i = 1}^m\mathbb I(f(\mathbf x_i) \neq y_i)$$
            $$E(f; D) = \int_{x \sim D}\mathbb I(f(\mathbf x_i) \neq y_i)p(\mathbf x)\mathrm d\mathbf x$$
        - 精度
            $$acc(f; D) = \frac1m\sum_{i = 1}^m\mathbb I(f(\mathbf x_i) = y_i) = 1 - E(f; D)$$
            $$acc(f; D) = \int_{x \sim D}\mathbb I(f(\mathbf x) = y)p(\mathbf x)\mathrm d\mathbf x = 1  - E(f; D)$$
3. 比较检验
4. 偏差与方差
    - 学习算法的期望预测：$\bar f(\mathbf x) = \mathbb E_D[f(\mathbf x, D)]$
    - 使用样本数相同的不同训练集产生的方差：$var(x) = \mathbb E_D[(f(\mathbf x; D) - \bar f(\mathbf x))^2]$
    - 噪声：$\epsilon^2 = \mathbb E_D[(y_D - y)^2]$
    - 偏差：$bias^2(\mathbf x) = (\bar f(\mathbf x) - y)^2$
    $$E(f; D) = \mathbb E_D[(f(\mathbf x; D) - y_D)^2] = \mathbb E_D[(f(\mathbf x; D) - \bar f(\mathbf x))^2] + \mathbb E_D[(\bar f(\mathbf x) - y)^2] + \mathbb E_D[(y_D - y)^2] = bias^2(\mathbf x) + var(\mathbf x) + \epsilon^2$$
5. 容量、过拟合和欠拟合
    1. 没有免费午餐定理
        - $\mathcal L_a$的“训练集外误差”：$E_{ote}(\mathcal L_a | X, f) = \sum_h\sum_{\mathbf x \in \mathcal X - X}P(x)\mathbb I(h(\mathbf x) \neq f(\mathbf x))P(h|X, \mathcal L_a)$
        $$\sum_fE_{ote}(\mathcal L_a | X, f) = \sum_f\sum_h\sum_{\mathbf x \in \mathcal X - X}P(x)\mathbb I(h(\mathbf x) \neq f(\mathbf x))P(h|X, \mathcal L_a) = 2^{|\mathcal X| - 1}\sum_{\mathbf x \in \mathcal X - X}P(\mathbf x) \Rightarrow E_{ote}(\mathcal L_a | X, f) = E_{ote}(\mathcal L_b | X, f)$$
    2. 正则化
        - 权重衰减：$J(\mathbf\omega) = \mathrm{MSE}_{train} + \lambda\mathbf\omega^\top\mathbf\omega$
        - 正则化项：$\Omega(\mathbf\omega) = \mathbf\omega^\top\mathbf\omega$
        1. [岭回归](ridge.ipynb)
        2. [Lasso回归](lasso.ipynb)
        3. [弹性网络](elastic_net.ipynb)
        4. [提前停止](early_stopping.py)
6. 最大似然估计
    $$\mathbf\theta_{\mathrm{ML}} = \argmax_{\mathbf\theta}p_{\mathrm{model}}(\mathbb X; \mathbf\theta) = \argmax_{\mathbf\theta}\prod_{i = 1}^m p_{\mathrm{model}}(\mathbf x^{(i)}; \mathbf\theta) = \argmax_{\mathbf\theta}\sum_{i = 1}^m \log p_{\mathrm{model}}(\mathbf x^{(i)}; \mathbf\theta) = \argmax_{\mathbf\theta}\mathbb E_{\mathbf x \sim \hat p_{\mathrm{data}}}  \log p_{\mathrm{model}}(\mathbf x; \mathbf\theta)$$
    1. 条件对数似然
        $$\mathbf\theta_{\mathrm{ML}} = \argmax_{\mathbf\theta}P(\mathbf Y | \mathbf X; \mathbf\theta) = \argmax_{\mathbf\theta}\sum_{i = 1}^m\log P(\mathbf y^{(i)} | \mathbf x^{(i)}; \mathbf\theta)$$
    2. 最大似然的性质
        - 当样本数目$m \rightarrow \infty$时，就收敛而言是最好的渐进估计
        - 在合适的条件下，最大似然估计具有一致性：
            - 真实分布$p_{\mathrm{data}}$必须在模型族$p_{\mathrm{model}}(·; \mathbf\theta)$中
            - 真实分布$p_{\mathrm{data}}$必须刚好对应一个$\mathbf\theta$值
7. 贝叶斯统计
    $$p(\mathbf\theta | x^{(1)}, ..., x^{(m)}) = \frac{p(x^{(1)}, ..., x^{(m)} | \mathbf\theta)p(\mathbf\theta)}{p(x^{(1)}, ..., x^{(m)})} \Rightarrow p(x^{(m + 1)} | x^{(1)}, ..., x^{(m)}) = \int p(x^{(m + 1)} | \mathbf\theta)p(\mathbf\theta | x^{(1)}, ..., x^{(m)})\mathrm d\mathbf\theta$$
    - 不像最大似然方法预测时使用$\mathbf\theta$的点估计，贝叶斯方法使用$\mathbf\theta$的全分布
    - 贝叶斯方法和最大似然方法的第二个最大区别是由贝叶斯先验分布造成的
8. 最大后验（MAP）估计
    $$\mathbf\theta_{\mathrm{MAP}} = \argmax_{\mathbf\theta}p(\mathbf\theta | \mathbf x) = \argmax_{\mathbf\theta}\log p(\mathbf x | \mathbf\theta) + \log p(\mathbf\theta)$$