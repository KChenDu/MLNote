1. 评估方法
    1. 留出法：直接将数据集$D$划分为两个互斥的集合，其中一个集合作为训练集$S$，另一个作为测试集$T$，即$D = S \cup T$，$S \cap T = \empty$。在$S$上训练出模型后，用$T$来评估其测试误差，作为对泛化误差的估计。
        - 训练/测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程引入额外的偏差而对最终结果产生影响
        - 即使在给定训练/测试集的样本比例后，仍存在多种划分方式对初始数据集$D$进行分割
    2. 交叉验证法：先将数据集$D$划分为$k$个大小相似的互斥子集，即$D = D_1 \cup D_2 \cup ... \cup D_k$。每个子集$D_i$都尽可能保持数据分布的一致性，即从$D$中通过分层采样得到。然后，每次用$k - 1$个子集的并集作为训练集，余下的那个子集作为测试集；这样就可获得$k$组训练/测试集，从而可进行$k$次训练和测试，最终返回的是这$k$个测试结果的均值。
        - 与留出法相似，将数据集$D$划分为$k$个子集同样存在多种划分方式
    3. 自助法：給定包含m个样本的数据集$D$，我们对它进行采样产生数据集$D^\prime$：每次随机从$D$中挑选一个样本，将其拷贝放入$D^\prime$，然后再将该样本放回初始数据集$D$中，使得该样本在下次采样时仍有可能被采到；这个过程重复执行$m$次后，我们就得到了包含$m$个样本的数据集$D'$，这就是自助采样的结果。
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
    2. 查准率、查全率与F1
        - 查准率：$P = \frac{TP}{TP + FP}$
        - 查全率：$R = \frac{TP}{TP + FN}$

        查准率和查全率是一对矛盾的度量。一般来说，查准率高时，查全率往往偏低；而查全率高时，查准率往往偏低

        在很多情形下，我们可根据学习器的预测结果对样例进行排序，排在前面的是学习器认为“最可能”是正例的样本，排在最后的则是学习器认为“最不可能”是正例的样本。按此顺序逐个把样本作为正例进行预测，则每次可以计算出当前的查全率、查准率。以查准率为纵轴、查全率为横轴作图，就得到了查准率-查全率曲线，简称“P-R曲线”，显示该曲线的图称为“P-R图”。P-R图直观地显示出学习器在样本总体上的查全率、查准率。在进行比较时，若一个学习器的P-R曲线被另一个学习器的曲线完全“包住”，则可断言后者的性能优于前者。然而，在很多情形下，人们往往仍希望把学习器A与B比出个高低。这时一个比较合理的判据是比较P-R曲线下面积的大小，它在一定程度上表征了学习器在查准率和查全率上取得相对“双高”的比例

        - 平衡点：“查准率 = 查全率”时的取值
        - F1度量：$F1 = \frac{2 \times P \times R}{P + R} = \frac{2 \times TP}{样例总数 + TP - TN}$
        
        
        F1度量的一般形式——$F_\beta$能让我们表达出对查准率/查全率的不同偏好，它定义为$F_\beta = \frac{(1 + \beta^2) \times P \times R}{(\beta^2 \times P) + R}$

        很多时候我们有多个二分类混淆矩阵，例如进行多次训练/测试，每次得到一个混淆矩阵；或是在多个数据集上进行训练/测试，希望估计算法的“全局”性能；甚或是执行多分类任务，每两两类别的组合都对应一个混淆矩阵；……总之，我们希望在n个二分类混淆矩阵上综合考察查准率和查全率
        $$
        \begin{cases}
            \text{macro-}P = \frac1n\sum_{i = 1}^nP_i \\
            \text{macro-}R = \frac1n\sum_{i = 1}^nR_i
        \end{cases}
        \Rightarrow \text{macro-}F1 = \frac{2 \times \text{macro-}P \times \text{macro-}R}{\text{macro-}P + \text{macro-}R}$$
        $$
        \begin{cases}
            \text{micro-}P = \frac{\overline{TP}}{\overline{TP} + \overline{FP}} \\
            \text{micro-}R = \frac{\overline{TP}}{\overline{TP} + \overline{FN}}
        \end{cases}
        \Rightarrow \text{micro-}F1 = \frac{2 \times \text{micro-}P \times \text{micro-}R}{\text{micro-}P + \text{micro-}R}$$
    3. ROC与AUC
        
        我们根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算出两个重要量的值，分别以它们为横、纵坐标作图,就得到了“ROC曲线”与P-R曲线使用查准率、查全率为纵、横轴不同，ROC曲线的纵轴是“真正例率”（简称TPR），横轴是“假正例率”（简称FPR），两者分别定义为
        $$\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$$
        $$\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{TN} + \mathrm{FP}}$$
        进行学习器的比较时，与P-R图相似，若一个学习器的ROC曲线被另一个学习器的曲线完全“包住”，则可断言后者的性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性地断言两者孰优孰劣。此时如果一定要进行比较，则较为合理的判据是比较ROC曲线下的面积，即AUC
        $$\mathrm{AUC} = \frac12\sum_{i = 1}^{m - 1}(x_{i + 1} - x_i)(y_i + y_{i + i})$$
        形式化地看，AUC考虑的是样本预测的排序质量，因此它与排序误差有紧密联系。给定$m^+$个正例和$m^-$个反例，令$D^+$和$D^-$分别表示正、反例集合，则排序“损失”定义为$l_\mathrm{rank} = \frac1{m^+m^-}\sum_{\mathbf x^+ \in D^+}\sum_{\mathbf x^- \in D^-}(\mathbb I(f(\mathbf x^+) < f(\mathbf x^-)) + \frac12\mathbb I(f(\mathbf x^+) = f(\mathbf x^-))) \Rightarrow \mathrm{AUC} = 1 - l_\mathrm{rank}$
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
6. 贝叶斯决策论
    
    假设有$N$种可能的类别标记，即$\mathcal Y = {c_1, c_2, ..., c_N}$，$\lambda_{ij}$是将一个真实标记为$c_j$的样本误分类为$c_i$所产生的损失
    - 期望损失：$R(c_i | \mathrm x) = \sum_{j = 1}^N\lambda_{ij}P(c_j | \mathbf x)$
    - 总体风险：$R(h) = \mathbb E_\mathbf x[R(h(\mathbf x) | \mathbf x)] \Rightarrow h^*(\mathbf x) = \argmin_{c \in \mathcal y} R(c | \mathbf x)$
    $$\lambda_{ij} =
    \begin{cases}
        0\text{, if }i = j \\
        1\text{, otherwise}
    \end{cases}
    \Rightarrow R(c | \mathbf x) = 1 - P(c | \mathbf x) \Rightarrow h^*(\mathbf x) = \argmax_{c \in \mathcal Y}P(c | \mathbf x)$$
    $$P(c | \mathbf x) = \frac{P(\mathbf x, c)}{P(\mathbf x)} = \frac{P(c)P(\mathbf x | c)}{P(\mathbf x)}$$
    - $P(c)$是类先验概率；$P(\mathbf x | c)$是样本$\mathbf x$相对于类标记$c$的类条件概率；$P(\mathbf x)$是用于归一化的证据因子
7. 最大似然估计
    $$P(D_c | \theta_c) = \prod_{\mathbf x \in D_c}P(\mathbf x | \mathbf\theta_c) \Rightarrow\mathbf\theta_{\mathrm{ML}} = \argmax_{\mathbf\theta}p_{\mathrm{model}}(\mathbb X; \mathbf\theta) = \argmax_{\mathbf\theta}\prod_{i = 1}^m p_{\mathrm{model}}(\mathbf x^{(i)}; \mathbf\theta) = \argmax_{\mathbf\theta}\sum_{i = 1}^m \log p_{\mathrm{model}}(\mathbf x^{(i)}; \mathbf\theta) = \argmax_{\mathbf\theta}\mathbb E_{\mathbf x \sim \hat p_{\mathrm{data}}}  \log p_{\mathrm{model}}(\mathbf x; \mathbf\theta)$$
    1. 对数似然：$LL(\theta_c) = \log P(D_c | \mathbf\theta_c) = \sum_{x \in D_c}\log P(\mathbf x | \mathbf\theta_c) \Rightarrow \hat{\mathbf\theta}_c = \argmax_{\mathbf\theta_c} LL(\mathbf\theta_c)$
    2. 条件对数似然：$\mathbf\theta_{\mathrm{ML}} = \argmax_{\mathbf\theta}P(\mathbf Y | \mathbf X; \mathbf\theta) = \argmax_{\mathbf\theta}\sum_{i = 1}^m\log P(\mathbf y^{(i)} | \mathbf x^{(i)}; \mathbf\theta)$
    3. 最大似然的性质
        1. 当样本数目$m \rightarrow \infty$时，就收敛而言是最好的渐进估计
        2. 在合适的条件下，最大似然估计具有一致性：
            - 真实分布$p_{\mathrm{data}}$必须在模型族$p_{\mathrm{model}}(·; \mathbf\theta)$中
            - 真实分布$p_{\mathrm{data}}$必须刚好对应一个$\mathbf\theta$值
    - 在连续属性情形下，假设概率密度函数$p(\mathbf x | c) \sim \mathcal N(\mathbf \mu_c, \mathbf\sigma_c^2)$，则参数$\mathbf \mu_c$和$\mathbf\sigma_c^2$的极大似然估计为
        $$\sum_{i = 1}m\log p(y^{(i)} | \mathbf{x}^{(i)}; \mathbf\theta) = -m\log\sigma - \frac m2\log(2\pi) - \sum_{i = 1}^m\frac{||\hat y^{(i)} - y^{(i)}||^2}{2\sigma^2} \Rightarrow
        \begin{cases}
            \mathbf\mu_c = \frac1{|D_c|}\sum_{\mathbf x \in D_c}\mathbf x \\
            \mathbf\sigma_c^2 = \frac1{|D_c|}\sum_{\mathbf x \in D_c}(\mathbf x - \hat{\mathbf\mu_c})(\mathbf x - \hat{\mathbf\mu_c})^\top \\
            \mathrm{MSE}_{\mathrm{train}} = \frac1m\sum_{i = 1}^m||\hat y^{(i)} - y^{(i)}||^2
        \end{cases}
        $$
8. 贝叶斯统计
    $$p(\mathbf\theta | x^{(1)}, ..., x^{(m)}) = \frac{p(x^{(1)}, ..., x^{(m)} | \mathbf\theta)p(\mathbf\theta)}{p(x^{(1)}, ..., x^{(m)})} \Rightarrow p(x^{(m + 1)} | x^{(1)}, ..., x^{(m)}) = \int p(x^{(m + 1)} | \mathbf\theta)p(\mathbf\theta | x^{(1)}, ..., x^{(m)})\mathrm d\mathbf\theta$$
    - 不像最大似然方法预测时使用$\mathbf\theta$的点估计，贝叶斯方法使用$\mathbf\theta$的全分布
    - 贝叶斯方法和最大似然方法的第二个最大区别是由贝叶斯先验分布造成的
9. 最大后验（MAP）估计
    $$\mathbf\theta_{\mathrm{MAP}} = \argmax_{\mathbf\theta}p(\mathbf\theta | \mathbf x) = \argmax_{\mathbf\theta}\log p(\mathbf x | \mathbf\theta) + \log p(\mathbf\theta)$$
10. [随机梯度下降](../DL/sgd.ipynb)

    在算法的每一步，我们从训练集中均匀抽出一**小批量**样本$\mathbb B = \{\mathbf x^{(1)}, \dots, \mathbf x^{(m^\prime)}\}$。梯度的估计可以表示成$\mathbf g = \frac1{m^\prime}\nabla_{\mathbf\theta}\sum_{i = 1}^{m^\prime}L(\mathbf x^{(i)}, y^{(i)}, \mathbf\theta)$

    然后，随机梯度下降算法使用如下的梯度下降估计：$\mathbf\theta \leftarrow \mathbf\theta - \epsilon\mathbf g$

[返回](../readme.md)