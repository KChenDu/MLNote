2. 神经网络优化中的挑战
    1. 病态

        病态体现在随机梯度下降会“卡”在某些情况，此时即使很小的更新步长也会增加代价函数
        
        代价函数的二阶泰勒级数展开预测梯度下降中的$−ϵ\mathbf g$会增加$\frac12\epsilon^2\mathbf g^\top T\mathbf g - \epsilon\mathbf g^\top\mathbf g$到代价中。当$\frac12\epsilon^2\mathbf g^\top T\mathbf g$超过$\epsilon\mathbf g^\top\mathbf g$时，梯度的病态会成为问题
    2. 局部极小值：由于**模型可辨识性**问题，神经网络和任意具有多个等效参数化潜变量的模型都会具有多个局部极小值。如果一个足够大的训练集可以唯一确定一组模型参数，那么该模型被称为可辨认的。带有潜变量的模型通常是不可辨认的，因为通过相互交换潜变量我们能得到等价的模型
    3. 高原、鞍点和其他平坦区域
    4. [悬崖和梯度爆炸](gradient_clipping.py)：多层神经网络通常存在像悬崖一样的斜率较大区域。这是由于几个较大的权重相乘导致的。遇到斜率极大的悬崖结构时，梯度更新会很大程度地改变参数值，通常会完全跳过这类悬崖结构
    5. 长期依赖：当计算图变得极深时，神经网络优化算法会面临的另外一个难题就是长期依赖问题——由于变深的结构使模型丧失了学习到先前信息的能力，让优化变得极其困难
3. 基本算法
    1. [随机梯度下降](sgd.ipynb)
        - 保证SGD收敛的一个充分条件是$\sum_{k = 1}^\infty\epsilon_k = \infty$且$\sum_{k = 1}^\infty\epsilon_k^2 < \infty$
        
            实践中，一般会线性衰减学习率直到第$\tau$次迭代：
            $$\epsilon_k = (1 - \alpha)\epsilon_0 + \alpha\epsilon_\tau$$
    2. [动量](momentum_optimization.py)
        ![SGD with Momentum](SGDwithMomentum.png "SGD with Momentum")
        从形式上看，动量算法引入了变量$\mathbf v$充当速度角色——它代表参数在参数空间移动的方向和速率。速度被设为负梯度的指数衰减平均。更新规则如下：
        $$\mathbf v \leftarrow \alpha\mathbf v - \epsilon\nabla_{\mathbf\theta}(\frac1m\sum_{i = 1}^mL(f(\mathbf x^{(i)}; \mathbf\theta), \mathbf y^{(i)})) \\
        \mathbf\theta \leftarrow \mathbf\theta + \mathbf v$$
        - 步长大小：$\frac{\epsilon\|\mathbf g\|}{1 - \alpha}$
    3. [Nesterov动量](nesterov.py)
        $$\mathbf v \leftarrow \alpha\mathbf v - \epsilon\nabla_{\mathbf\theta}(\frac1m\sum_{i = 1}^mL(f(\mathbf x^{(i)}; \mathbf\theta + \alpha\mathbf v), \mathbf y^{(i)})) \\
        \mathbf\theta \leftarrow \mathbf\theta + \mathbf v$$
        Nesterov动量和标准动量之间的区别体现在梯度计算上。Nesterov动量中，梯度计算在施加当前速度之后。因此，Nesterov动量可以解释为往标准动量方法中添加了一个校正因子
        ![Nesterov](Nesterov.png "Nesterov")
        在凸批量梯度的情况下，Nesterov动量将额外误差收敛率从$O(1 / k)$（k 步后）改进到$O(1/k^2)$。可惜，在随机梯度的情况下，Nesterov动量没有改进收敛率
4. 参数初始化策略

	也许完全确知的唯一特性是初始参数需要在不同单元间‘‘破坏对称性’’。如果具有相同激活函数的两个隐藏单元连接到相同的输入，那么这些单元必须具有不同的初始参数。如果它们具有相同的初始参数，然后应用到确定性损失和模型的确定性学习算法将一直以相同的方式更新这两个单元。每个单元计算不同函数的目标促使了参数的随机初始化
	
	通常情况下，我们可以为每个单元的偏置设置启发式挑选的常数，仅随机初始化权重。额外的参数（例如用于编码预测条件方差的参数）通常和偏置一样设置为启发式选择的常数

	更大的初始权重具有更强的破坏对称性的作用，有助于避免冗余的单元。它们也有助于避免在每层线性成分的前向或反向传播中丢失信号——矩阵中更大的值在矩阵乘法中有更大的输出。如果初始权重太大，那么会在前向传播或反向传播中产生爆炸的值。在循环网络中，很大的权重也可能导致**混沌**（对于输入中很小的扰动非常敏感，导致确定性前向传播过程表现随机）

	有些启发式方法可用于选择权重的初始大小。一种初始化$m$个输入和$n$输出的全连接层的权重的启发式方法是从分布$W_{i, j} \sim U(−\sqrt{\frac1m}; \sqrt{\frac1m})$中采样权重，建议使用**标准初始化**$W_{i, j} \sim U(−\sqrt{\frac6{m + n}}; \sqrt{\frac6{m + n}})$

	后一种启发式方法初始化所有的层，折衷于使其具有相同激活方差和使其具有相同梯度方差之间。这假设网络是不含非线性的链式矩阵乘法，据此推导得出。现实的神经网络显然会违反这个假设，但很多设计于线性模型的策略在其非线性对应中的效果也不错

	在实践中，我们通常需要将权重范围视为超参数，其最优值大致接近，但并不完全等于理论预测。

	数值范围准则的一个缺点是，设置所有的初始权重具有相同的标准差，例如$\frac1{\sqrt m}$，会使得层很大时每个单一权重会变得极其小。
	- **稀疏初始化**：每个单元初始化为恰好有$k$个非零权重。这个想法保持该单元输入的总数量独立于输入数目$m$，而不使单一权重元素的大小随$m$缩小。稀疏初始化有助于实现单元之间在初始化时更具多样性。但是，获得较大取值的权重也同时被加了很强的先验。因为梯度下降需要很长时间缩小‘‘不正确’ 的大值，这个初始化方案可能会导致某些单元出问题，例如maxout单元有几个过滤器，互相之间必须仔细调整
	
	计算资源允许的话，将每层权重的初始数值范围设为超参数通常是个好主意，使用超参数搜索算法，如随机搜索，挑选这些数值范围。是否选择使用密集或稀疏初始化也可以设为一个超参数。作为替代，我们可以手动搜索最优初始范围。一个好的挑选初始数值范围的经验法则是观测单个小批量数据上的激活或梯度的幅度或标准差。如果权重太小，那么当激活值在小批量上前向传播于网络时，激活值的幅度会缩小。通过重复识别具有小得不可接受的激活值的第一层，并提高其权重，最终有可能得到一个初始激活全部合理的网络。如果学习在这点上仍然很慢，观测梯度的幅度或标准差可能也会有所帮助

	设置偏置为零通常在大多数权重初始化方案中是可行的。存在一些我们可能设置偏置为非零值的情况：
	- 如果偏置是作为输出单元，那么初始化偏置以获取正确的输出边缘统计通常是有利的
	- 有时，我们可能想要选择偏置以避免初始化引起太大饱和
	- 有时，一个单元会控制其他单元能否参与到等式中。在这种情况下，我们有一个单元输出$u$，另一个单元$h \in [0, 1]$，那么我们可以将$h$视作门，以决定$uh \approx 1$还是$uh \approx 0$。在这种情形下，我们希望设置偏置$h$，使得在初始化的大多数情况下$h \approx 1$。否则，$u$没有机会学习

	另一种常见类型的参数是方差或精确度参数。例如，我们用以下模型进行带条件方差估计的线性回归：$p(y | \mathbf x) = \mathcal N(y | \mathbf w^\top\mathbf x + b, 1/\beta)$，其中$\beta$是精确度参数

	除了这些初始化模型参数的简单常数或随机方法，还有可能使用机器学习初始化模型参数
5. 自适应学习率算法
    1. [AdaGrad](adagrad.py)
        - 独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平方值总和的平方根
        ![AdaGrad](Adagrad.png "AdaGrad")
    2. [RMSProp](rmsprop.py)
        - 改变梯度积累为指数加权的移动平均
       ![RMSProp](RMSProp.png "RMSProp")
    3. [Adam](adam_optimization.ipynb)
        - 首先，在Adam中，动量直接并入了梯度一阶矩（指数加权）的估计。将动量加入RMSProp最直观的方法是将动量应用于缩放后的梯度。其次，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩的估计
       ![Adam](Adam.png "Adam")
6. 二阶近似方法
    $$J(\mathbf\theta) = \mathbb E_{\mathbf x, y \sim \hat p_{\mathrm{data}}(\mathbf x, y)}[L(f(\mathbf x; \mathbf\theta), y)] = \frac1m\sum_{i= 1}^mL(f(\mathbf x^{(i)}; \mathbf\theta), y^{(i)})$$
    1. 牛顿法
        $$J(\mathbf\theta) \approx J(\mathbf\theta^{(0)}) + (\mathbf\theta - \mathbf\theta^{(0)})^\top\nabla_{\mathbf\theta}J(\mathbf\theta^{(0)}) + \frac12(\mathbf\theta - \mathbf\theta^{(0)})^\top H(\mathbf\theta^{(0)})(\mathbf\theta - \mathbf\theta^{(0)}) \Rightarrow \mathbf\theta^* = \mathbf\theta_0 - H^{-1}\nabla_{\mathbf\theta}J(\mathbf\theta)$$
        在深度学习中，目标函数的表面通常非凸（有很多特征），如鞍点。因此使用牛顿法是有问题的。如果Hessian矩阵的特征值并不都是正的，例如，靠近鞍点处，牛顿法实际上会导致更新朝错误的方向移动。这种情况可以通过正则化Hessian矩阵来避免。常用的正则化策略包括在Hessian矩阵对角线上增加常数$\alpha$。正则化更新变为$\mathbf\theta^* = \mathbf\theta_0 - (H(f(\mathbf\theta_0)) + \alpha I)^{-1}\nabla_{\mathbf\theta}J(\mathbf\theta)$
       ![牛顿法](Newton'sMethod.png "牛顿法")

[返回](readme.md)