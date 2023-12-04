1. [基于梯度的学习](the_perceptron.ipynb)
    1. 代价函数
        1. 使用最大似然学习条件分布
            $$J(\mathbf\theta) = -\mathbb E_{\mathbf x, \mathbf y \sim \hat p_{\mathrm{data}}}\log p_{model}(\mathbf y | \mathbf x)$$
            $$p_{\mathrm{model}}(\mathbf y | \mathbf x) = \mathcal N(\mathbf y; f(\mathbf x, \mathbf\theta), I) \Rightarrow J(\mathbf\theta) = \frac12\mathbb E_{\mathbf x, \mathbf y \sim \hat p_{\mathrm{data}}}\|\mathbf y - f(\mathbf x; \mathbf\theta)\|^2 + \mathrm{const}$$
            使用最大似然来导出代价函数的方法的一个优势是，它减轻了为每个模型设计代价函数的负担。明确一个模型$p(\mathbf y | \mathbf x)$则自动地确定了一个代价函数$\log p(\mathbf y | \mathbf x)$
        2. 学习条件统计量
            $$
            \begin{cases}
                f^* = \argmin_f\mathbb E_{\mathbf x, \mathbf y \sim p_{\mathrm{data}}}\|\mathbf y - f(\mathbf x)\|^2 \\
                f^* = \argmin_f\mathbb E_{\mathbf x, \mathbf y \sim p_{\mathrm{data}}}\|\mathbf y - f(\mathbf x)\|_1
            \end{cases}
            \Rightarrow f^*(\mathbf x) = \mathbb E_{\mathbf y \sim p_{\mathrm{data}}(\mathbf y | \mathbf x)}[y]$$
            可惜的是，均方误差和平均绝对误差在使用基于梯度的优化方法时往往成效不佳。一些饱和的输出单元当结合这些代价函数时会产生非常小的梯度
    2. 输出单元
        1. 用于高斯输出分布的线性单元
        2. 用于Bernoulli输出分布的sigmoid单元
        3. 用于Multinoulli输出分布的softmax单元
2. 隐藏单元
    1. 整流线性单元及其[扩展](nonsaturating_activation_functions.ipynb)

        整流线性单元使用激活函数$g(z) = \max \{0, z\}$

        整流线性单元易于优化，因为它们和线性单元非常类似。线性单元和整流线性单元的唯一区别在于整流线性单元在其一半的定义域上输出为零。这使得只要整流线性单元处于激活状态，它的导数都能保持较大。它的梯度不仅大而且一致。整流操作的二阶导数几乎处处为0，并且在整流线性单元处于激活状态时，它的一阶导数处处为1。这意味着相比于引入二阶效应的激活函数来说，它的梯度方向对于学习来说更加有用

        整流线性单元通常作用于仿射变换之上：$\mathbf h = g(W^\top\mathbf x + \mathbf b)$

        整流线性单元的一个缺陷是它们不能通过基于梯度的方法学习那些使它们激活为零的样本。整流线性单元的各种扩展保证了它们能在各个位置都接收到梯度。

        整流线性单元的三个扩展基于当$z_i < 0$时使用一个非零的斜率$\alpha_i$：$h_i = g(\mathbf z; \mathbf\alpha)_i = \max(0; z_i) + \alpha_i \min(0; zi)$
        - 绝对值整流：$\alpha_i = -1 \Rightarrow g(z) = |z|$
        - [渗漏整流线性单元](leaky_ReLU.ipynb)
        - 参数化整流线性单元：将$\alpha_i$作为学习的参数
        - maxout单元：$g(\mathbf z)_i = \max_{j \in \mathbb G^{i}} z_j$

        maxout单元可以学习具有多达$k$段的分段线性的凸函数
    2. logistic sigmoid与双曲正切函数
        - logistic sigmoid激活函数：$g(z) = \sigma(z)$
        - 双曲正切激活函数：$g(z) = \tanh(z)$

        我们已经看过sigmoid单元作为输出单元用来预测二值型变量取值为1的概率。与分段线性单元不同，sigmoid单元在其大部分定义域内都饱和——当$z$取绝对值很大的正值时，它们饱和到一个高值，当$z$取绝对值很大的负值时，它们饱和到一个低值，并且仅仅当$z$接近0时它们才对输入强烈敏感。sigmoid单元的广泛饱和性会使得基于梯度的学习变得非常困难
    3. 其他隐藏单元
        - 其中一种是完全没有激活函数$g(z)$。也可以认为这是使用单位函数作为激活函数的情况
        - softmax单元很自然地表示具有$k$个可能值的离散型随机变量的概率分布，所以它们可以用作一种开关
        - 径向基函数：$h_i = \exp(-\frac1{\sigma_i^2}\|W_{:, i} - \mathbf x\|^2)$这个函数在$x$接近模板$W_{:, i}时更加活跃。因为它对大部分$x$都饱和到0,因此很难优化
        - softplus函数：$g(a) = \zeta(a) = \log(1 + e^a)$这是整流线性单元的平滑版本
        - 硬双曲正切函数：它的形状和$\tanh$以及整流线性单元类似，但是不同于后者，它是有界的$g(a) = \max(-1, \min(1, a))$
3. 架构设计

    大多数神经网络被组织成称为层的单元组
    $$\mathbf h^{(1)} = g^{(1)}(W^{(1)}\mathbf x + \mathbf b^{(1)}) \\
    \mathbf h^{(2)} = g^{(2)}(W^{(2)}\mathbf h + \mathbf b^{(2)})$$
    1. 万能近似性质和深度

        一个前馈神经网络如果具有线性输出层和至少一层具有任何一种“挤压”性质的激活函数（例如logistic sigmoid激活函数）的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的Borel可测函数

        存在一些函数族能够在网络的深度大于某个值$d$时被高效地近似，而当深度被限制到小于或等于$d$时需要一个远远大于之前的模型

        Montufar的主要定理指出，具有$d$个输入、深度为$l$、每个隐藏层具有$n$个单元的深度整流网络可以描述的线性区域的数量是$O({n \choose d}^{d(l - 1)}n^d)$意味着，这是深度$l$的指数级。在每个单元具有$k$个过滤器的maxout网络中，线性区域的数量是$O(k^{(l - 1) + d})$
    2. 其他架构上的考虑

        一般来说，层不需要连接在链中，尽管这是最常见的做法。许多架构构建了一个主链，但随后又添加了额外的架构特性，例如从层$i$到层$i + 2$或者更高层的跳跃连接。这些跳跃连接使得梯度更容易从输出层流向更接近输入的层

        许多专用网络具有较少的连接，使得输入层中的每个单元仅连接到输出层单元的一个小子集。这些用于减少连接数量的策略减少了参数的数量以及用于评估网络的计算量，但通常高度依赖于问题

[返回](readme.md)