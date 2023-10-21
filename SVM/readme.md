1. 间隔与支持向量
    $$\mathbf w^\top\mathbf x + b = 0$$
    样本空间中任意点$\mathbf x$到超平面$(\mathbf w, b)$的的距离可写为$r = \frac{|\mathbf w^\top\mathbf x + b|}{\|\mathbf w\|}\Rightarrow \gamma = \frac{2}{\|\mathbf w\|}$
    
    两个异类支持向量到超平面的距离之和为$\gamma = \frac{2}{\|w\|}$
    $$\max_{\mathbf w, b}\frac{2}{\|\mathbf w\|}\text{ s.t. }y_i(\mathbf w^\top\mathbf x_i + b) \ge 1 \Leftrightarrow \min_{\mathbf w, b}\frac{\|\mathbf w\|^2}2\text{ s.t. }y_i(\mathbf w^\top\mathbf x_i + b) \ge 1$$
2. 对偶问题
	$$L(\mathbf w, b, \mathbf\alpha) = \frac12\|\mathbf w\|^2 + \sum_{i = 1}^m\alpha_i(1 - y_i(\mathbf w^\top\mathbf x_i + b)) \Rightarrow \begin{cases} \mathbf w = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i \\ 0 = \sum_{i = 1}^m\alpha_iy_i \end{cases} \Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\mathbf x_i^\top\mathbf x_j\text{ s.t. }\sum_{i = 1}^ma_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m \Rightarrow f(\mathbf x) = \mathbf w^\top\mathbf x + b = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i^\top\mathbf x + b$$
	- KTT条件：$\begin{cases}\alpha_i \ge 0 \\ y_if(\mathbf x_i) - 1 \ge 0 \\ \alpha_i(y_if(\mathbf x_i) - 1) = 0\end{cases}$
	
	SMO不断执行如下两个步骤直至收敛：
		
	- 选取一对需更新的变量$\alpha_i$和$\alpha_j$
	- 固定$\alpha_i$和$\alpha_j$以外的参数，求解式获得更新后的$\alpha_i$和$\alpha_j$
	$$\alpha_iy_i + \alpha_jy_j = c\text{, }\alpha_i \ge 0\text{, }\alpha_j \ge 0\text{, }c = -\sum_{k \neq i, j}\alpha_ky_k$$
	$$y_s(\sum_{i \in S}\alpha_iy_i\mathbf x_i^\top\mathbf x_s + b) = 1 \Rightarrow b = \frac1{|S|}\sum_{s \in S}(1 / y_s - \sum_{i \in S}\alpha_iy_i\mathbf x_i^\top\mathbf x_s)$$
3. [核函数](soft_margin_classification.ipynb)
	$$f(x) = \mathbf w^\top\phi(\mathbf x) + b$$
	$$\min_{\mathbf w, b}\frac{\|\mathbf w\|^2}2\text{ s.t. }y_i(\mathbf w^\top\phi(\mathbf x_i) + b) \ge 1\text{, }i = 1, 2, ..., m \Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\phi(\mathbf x_i)^\top\phi(\mathbf x_j)\text{ s.t. }\sum_{i = 1}^m\alpha_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m$$
	$$\kappa(\mathbf x_i, \mathbf x_j) = \langle\phi(\mathbf x_i), \phi(\mathbf x_j)\rangle = \phi(\mathbf x_i)^\top\phi(x_j) \Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\kappa(\mathbf x_i, \mathbf x_j)\text{ s.t. }\sum_{i = 1}^m\alpha_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m \Rightarrow f(\mathbf x) = \mathbf w^\top\mathbf \phi(x) + b = \sum_{i = 1}^m\alpha_iy_i\phi(\mathbf x_i)^\top\phi(\mathbf x) + b = \sum_{i = 1}^m\alpha_iy_i\kappa(\mathbf x, \mathbf x_i) + b$$
	- 核函数：令$\mathcal X$为输入空间，$\kappa(· | ·)$是定义在$\mathcal X \times \mathcal X$上的对称函数，则$\kappa$是核函数当且仅当对于任意数据$D = {x_1, x_2, ..., x_m}$，“核矩阵”$K$总是半正定的：
		$$K = \begin{bmatrix} \kappa(\mathbf x_1, \mathbf x_1) & \dots & \kappa(\mathbf x_1, \mathbf x_j) & \dots & \kappa(\mathbf x_1, \mathbf x_m) \\ \vdots & \ddots & \vdots & \ddots & \vdots \\ \kappa(\mathbf x_i, \mathbf x_1) & \dots & \kappa(\mathbf x_i, \mathbf x_j) & \dots & \kappa(\mathbf x_i, \mathbf x_m) \\ \vdots & \ddots & \vdots & \ddots & \vdots \\ \kappa(\mathbf x_m, \mathbf x_1) & \dots & \kappa(\mathbf x_m, \mathbf x_j) & \dots & \kappa(\mathbf x_m, \mathbf x_m)\end{bmatrix}$$
	
	| 名称 | 表达式 | 参数 |
	| - | - | - |
	| 线性核 | $\kappa(\mathbf x_i, \mathbf x_j) = \mathbf x_i^\top\mathbf x_j$ ||
	| [多项式核](polynomial_kernel.ipynb) | $\kappa(\mathbf x_i, \mathbf x_j) = (\mathbf x_i^\top\mathbf x_j)^d$ | $d \ge 1$为多项式的次数 |
	| [高斯核](gaussian_rbf_kernel.ipynb) | $\kappa(\mathbf x_i, \mathbf x_j) = \exp(-\frac{\|\mathbf x_i - \mathbf x_j\|^2}{2\sigma^2})$ | $\sigma > 0$为高斯核的带宽 |
	| 拉普拉斯核 | $\kappa(\mathbf x_i, \mathbf x_j) = \exp(-\frac{\|\mathbf x_i - \mathbf x_j\|^2}{\sigma})$ | $\sigma > 0$ |
	| sigmoid核 | $\kappa(\mathbf x_i, \mathbf x_j) = \tanh(\beta\mathbf x_i^\top\mathbf x_j + \theta)$ | $\tanh$为双曲正切函数，$\beta > 0, \theta < 0$ |
	- 若$\kappa_1$和$\kappa_2$为核函数，则对于任意正数$\gamma_1$、$\gamma_2$，其线性组合$\kappa_1\gamma_1 + \gamma_2\kappa_2$也是核函数
	- 若$\kappa_1$和$\kappa_2$为核函数，则核函数的直积$\kappa_1\bigotimes\kappa_2(\mathbf x, \mathbf z) = \kappa_1(\mathbf x, \mathbf z)\kappa_2(\mathbf x, \mathbf z)$也是核函数
	- 若$\kappa_1$为核函数，则对于任意函数$g(x)$，$\kappa(\mathbf x, \mathbf z) = g(\mathbf x)\kappa_1(\mathbf x, \mathbf z)g(\mathbf z)$也是核函数
4. [软间隔与正则化](soft_margin_classification.ipynb)
5. 支持向量回归
6. 核方法
	
	**表示定理**：令$\mathbb H$为核函数$\kappa$对应的再生核希尔伯特空间，$\|h\|_{\mathbb H}$表示$H$空间中关于$h$的范数，对于任意单调递增函数$\Omega: [0, \infty] \rightarrow \mathbb R$和任意非负损失函数$l: \mathbb R \rightarrow [0, \infty]$，优化问题$\min_{h \in \mathbb H}F(h) = \Omega(\|h\|_{\mathbb H}) + l(h(\mathbf x_1), h(\mathbf x_2), ..., h(\mathbf x_m))$的解总可写为$h^\ast(\mathbf x) = \sum_{i = 1}^m\alpha_i\kappa(\mathbf x, \mathbf x_i)$
	$$h(\mathbf x) = \mathbf w^\top\phi(\mathbf x) = \sum_{i = 1}^m\alpha_i\kappa(\mathbf x, \mathbf x_i) \Rightarrow \mathbf w = \sum_{i = 1}^m\alpha_i\phi(\mathbf x_i)$$

[返回](../readme.md)