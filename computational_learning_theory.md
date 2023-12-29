1. 基础知识
    
    给定样例集$D = \{(\mathbf x_1, y_1), (\mathbf x_2, y_2), \dots, (\mathbf x_m, y_m)\}$, $\mathbf x_i \in \mathcal X$。假设$X$中的所有样本服从一个隐含未知的分布$\mathcal D$，$D$中所有样本都是独立地从这个分布上采样而得，即独立同分布样本。

    令$h$为从$\mathcal X$到$\mathcal Y$的一个映射，其泛化误差为
    $$E(h; \mathcal D) = P_{\mathbf x \sim \mathcal D}(h(\mathbf x) \neq y)$$
    $h$在$\mathcal D$上的经验误差为
    $$\hat E(h; \mathcal D) = \frac1m\sum_{i = 1}^m\mathbb I(h(\mathbf x_i) \neq y_i)$$
    - 令$E$为$E(h)$的上限，即$E(h) < \epsilon$；我们通常用$\epsilon$表示预先设定的学得模型所应满足的误差要求，亦称“误差参数”
    - 若$h$在数据集$D$上的经验误差为0，则称$h$与$D$一致，否则称其与$D$不一致

    对任意两个映射$h_1$，$h_2 \in \mathcal X \rightarrow \mathcal Y$，可通过其“不合”来度量它们之间的差别：$d(h_1, h_2) = P_{\mathbf x \sim \mathcal D}(h_1(\mathbf x) \neq h_2(\mathbf x))$

    我们会用到几个常用不等式：
    - Jensen不等式：对任意凸函数$f(x)$，有：$f(\mathbb E(x)) \leq \mathbb E(f(x))$
    - Hoeffding不等式：若$x_1, x_2, \dots, x_m$为$m$个独立随机变量，且满足$0 \leq x_i \leq 1$，则对任意$\epsilon > 0$，有
        $$P(\frac1m\sum_{i = 1}^mx_i - \frac1m\sum_{i = 1}^m\mathbb E(x_i) \ge \epsilon) \leq \exp(-2m\epsilon^2)$$
        $$P(|\frac1m\sum_{i = 1}^mx_i - \frac1m\sum_{i = 1}^m\mathbb E(x_i)| \ge \epsilon) \leq \exp(-2m\epsilon^2)$$
    - McDiarmid不等式：若$x_1, x_2, \dots, x_m$为$m$个独立随机变量，且对任意$1 \leq i \leq m$，函数$f$满足$\sup_{x_1, \dots, x_m, x_i'}|f(x_1, \dots, x_m) - f(x_1, \dots, x_{i - 1}, x_i', x_{i + 1}, \dots, x_m)| \leq c_i$，则对任意$\epsilon > 0$，有
        $$P(f(x_1, \dots, x_m) - \mathbb E(x_1, \dots, x_m) \ge \epsilon) \leq \exp(\frac{-2\epsilon^2}{\sum_ic_i^2})$$
        $$P(|f(x_1, \dots, x_m) - \mathbb E(x_1, \dots, x_m)| \ge \epsilon) \leq \exp(\frac{-2\epsilon^2}{\sum_ic_i^2})$$
2. PAC学习

    **PAC辨识**：对$0 < \epsilon$，$\delta < 1$，所有$c\ \in \mathcal C$和分布$\mathcal D$，若存在学习算法$\mathcal L$，其输出假设$h \in \mathcal H$满足$P(E(h) \leq \epsilon) \ge 1 - \delta$，则称学习算法$\mathcal L$能从假设空间$\mathcal H$中PAC辨识概念类$\mathcal C$
    
    **PAC可学习**：令$m$表示从分布$\mathcal D$中独立同分布采样得到的样例数目，$0 < \epsilon$，$\delta < 1$，对所有分布$\mathcal D$，若存在学习算法$\mathcal L$和多项式函数$\mathrm{poly}(., ., ., .)$，使得对于任何$m \ge \mathrm{poly}(1 / \epsilon, 1 / \delta, \mathrm{size}(\mathbf x), \mathrm{size}(c))$，$\mathcal L$能从假设空间$\mathcal H$中PAC辨识概念类$\mathcal C$，则称概念类$\mathcal C$对假设空间$\mathcal H$而言是PAC可学习的

    **PAC学习算法**：若学习算法$\mathcal L$使概念类$\mathcal C$为PAC可学习的，且$\mathcal L$的运行时间也是多项式函数$\mathrm{poly}(1 / \epsilon, 1 / \delta, \mathrm{size}(\mathbf x), \mathrm{size}(c))$，则称概念类$\mathcal C$是高效PAC可学习的，称$\mathcal L$为概念类$\mathcal C$的PAC学习算法

    **样本复杂度**：满足PAC学习算法$\mathcal L$所需的$m \ge \mathrm{poly}(1 / \epsilon, 1 / \delta, \mathrm{size}(\mathbf x), \mathrm{size}(c))$中最小的m，称为学习算法$\mathcal L$的样本复杂度
3. 有限假设空间
    1. 可分情形

        假定$h$的泛化误差大于$\epsilon$，对分布$\mathcal D$上随机采样而得的任何样例$(\mathbf x, y)$，有$P(h(\mathbf x) = y) = 1 - P(h(\mathbf x) \neq y) = 1 - E(h) < 1 - \epsilon \Rightarrow P((h(\mathbf x_1) = y_1) \land \dots \land (h(\mathbf x_m) = y_m)) < (1 - \epsilon)^m$

        我们事先并不知道学习算法$\mathcal L$会输出$\mathcal H$中的哪个假设，但仅需保证泛化误差大于$\epsilon$，且在训练集上表现完美的所有假设出现概率之和不大于$\delta$即可：$P(h \in \mathcal H: E(h) > \epsilon \land \hat E(h) = 0) < |\mathcal H|(1 - \epsilon)^m < |\mathcal H|e^{-m\epsilon} \leq \delta \Rightarrow m \ge \frac1\epsilon(\ln|\mathcal H| + \ln\frac1\delta)$
    2. 不可分情形

        **引理**：若训练集$D$包含$m$个从分布$\mathcal D$上独立同分布采样而得的样例，$0 < \epsilon < 1$，则对任意$h \in \mathcal H$，有
        $$P(\hat E(h) - E(h) \ge \epsilon) \leq \exp(-2m\epsilon^2)$$
        $$P(E(h) - \hat E(h) \ge \epsilon) \leq \exp(-2m\epsilon^2)$$
        $$P(|E(h) - \hat E(h)| \ge \epsilon) \leq 2\exp(-2m\epsilon^2)$$

        **推论**：若训练集$D$包含$m$个从分布$\mathcal D$上独立同分布采样而得的样例，$0 < \epsilon < 1$，则对任意$h \in \mathcal H$，式以至少$1 - \delta$的概率成立：$\hat E(h) - \sqrt{\frac{\ln(2 / \delta)}{2m}} \leq E(h) \leq \hat E(h) + \sqrt{\frac{\ln(2 / \delta)}{2m}}$

        **定理**：若$\mathcal H$为有限假设空间，$0 < \delta < 1$，则对任意$h \in \mathcal H$，有$P(|E(h) - \hat E(h)| \leq \sqrt{\frac{\ln|\mathcal H| + \ln(2 / \delta)}{2m}}) \ge 1 - \delta$

        **不可知PAC可学习**：令$m$表示从分布$\mathcal D$中独立同分布采样得到的样例数目，$0 < \epsilon, \delta < 1$，对所有分布$\mathcal D$，若存在学习算法$\mathcal L$和多项式函数$\mathrm{poly}(., ., ., .)$，使得对于任何$m \ge \mathrm{poly}(1 / \epsilon, 1 / \delta, \mathrm{size}(\mathbf x), \mathrm{size}(c))$，$\mathcal L$能从假设空间$\mathcal H$中输出满足式的假设$h$：$P(E(h) - \min_{h' \in \mathcal H}E(h') \leq \epsilon) \ge 1 - \delta$，则称假设空间$\mathcal H$是不可知PAC可学习的

5. Rademacher复杂度
    
    给定训练集$D = \{(\mathbf x_1, y_1), (\mathbf x_2, y_2), \dots, (\mathbf x_m, y_m)\}$，假设$h$的经验误差为$\hat E(h) = \frac1m\sum_{i = 1}^m\mathbb I(h(\mathbf x_i ) \neq y_i) = \frac1m\sum_{i = 1}^m\frac{1 - y_ih(\mathbf x_i)}2 = \frac12 - \frac1{2m}\sum_{i = 1}^my_ih(\mathbf x_i)$

[返回](readme.md)