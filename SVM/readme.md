1. 间隔与支持向量
    $$\mathbf w^\top\mathbf x + b = 0$$
    样本空间中任意点$\mathbf x$到超平面$(\mathbf w, b)$的的距离可写为$r = \frac{|\mathbf w^\top\mathbf x + b|}{||\mathbf w||}\Rightarrow \gamma = \frac{2}{||\mathbf w||}$
    
    两个异类支持向量到超平面的距离之和为$\gamma = \frac{2}{||w||}$
    $$\max_{\mathbf w, b}\frac{2}{||\mathbf w||}\text{ s.t. }y_i(\mathbf w^\top\mathbf x_i + b) \ge 1 \Leftrightarrow \min_{\mathbf w, b}\frac{||\mathbf w||^2}2\text{ s.t. }y_i(\mathbf w^\top\mathbf x_i + b) \ge 1$$
2. 对偶问题
	$$L(\mathbf w, b, \mathbf\alpha) = \frac12||\mathbf w||^2 + \sum_{i = 1}^m\alpha_i(1 - y_i(\mathbf w^\top\mathbf x_i + b)) \Rightarrow
	\begin{cases}
		\mathbf w = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i \\
		0 = \sum_{i = 1}^m\alpha_iy_i
	\end{cases}
	\Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\mathbf x_i^\top\mathbf x_j\text{ s.t. }\sum_{i = 1}^ma_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m \Rightarrow f(\mathbf x) = \mathbf w^\top\mathbf x + b = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i^\top\mathbf x + b$$
	- KTT条件：$\begin{cases}\alpha_i \ge 0; \\ y_if(\mathbf x_i) - 1 \ge 0 \\ \alpha_i(y_if(\mathbf x_i) - 1) = 0\end{cases}$
3. 核函数
	$$f(x) = \mathbf w^\top\phi(\mathbf x) + b$$
	$$\min_{\mathbf w, b}\frac{||\mathbf w||^2}2\text{ s.t. }y_i(\mathbf w^\top\phi(\mathbf x_i) + b) \ge 1\text{, }i = 1, 2, ..., m \Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\phi(\mathbf x_i)^\top\phi(\mathbf x_j)\text{ s.t. }\sum_{i = 1}^m\alpha_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m$$
	$$\kappa(\mathbf x_i, \mathbf x_j) = \langle\phi(\mathbf x_i), \phi(\mathbf x_j)\rangle = \phi(\mathbf x_i)^\top\phi(x_j) \Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\kappa(\mathbf x_i, \mathbf x_j)\text{ s.t. }\sum_{i = 1}^m\alpha_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m \Rightarrow f(\mathbf x) = \mathbf w^\top\mathbf \phi(x) + b = \sum_{i = 1}^m\alpha_iy_i\phi(\mathbf x_i)^\top\phi(\mathbf x) + b = \sum_{i = 1}^m\alpha_iy_i\kappa(\mathbf x, \mathbf x_i) + b$$
	- 核函数：令$\mathcal X$为输入空间，$\kappa(· | ·)$是定义在$\mathcal X \times \mathcal X$上的对称函数，则$\kappa$是核函数当且仅当对于任意数据$D = {x_1, x_2, ..., x_m}$，“核矩阵”$K$总是半正定的：
		$$K =
	\begin{bmatrix} 
	    \kappa(\mathbf x_1, \mathbf x_1) & \dots & \kappa(\mathbf x_1, \mathbf x_j) & \dots & \kappa(\mathbf x_1, \mathbf x_m) \\
	    \vdots & \ddots & \vdots & \ddots & \vdots \\
	    \kappa(\mathbf x_i, \mathbf x_1) & \dots & \kappa(\mathbf x_i, \mathbf x_j) & \dots & \kappa(\mathbf x_i, \mathbf x_m) \\
	    \vdots & \ddots & \vdots & \ddots & \vdots \\
	    \kappa(\mathbf x_m, \mathbf x_1) & \dots & \kappa(\mathbf x_m, \mathbf x_j) & \dots & \kappa(\mathbf x_m, \mathbf x_m)
    \end{bmatrix}
	$$
	- 若$\kappa_1$和$\kappa_2$为核函数，则对于任意正数$\gamma_1$、$\gamma_2$，其线性组合$\kappa_1\gamma_1 + \gamma_2\kappa_2$也是核函数
	- 若$\kappa_1$和$\kappa_2$为核函数，则核函数的直积$\kappa_1\bigotimes\kappa_2(\mathbf x, \mathbf z) = \kappa_1(\mathbf x, \mathbf z)\kappa_2(\mathbf x, \mathbf z)$也是核函数
	- 若$\kappa_1$为核函数，则对于任意函数$g(x)$，$\kappa(\mathbf x, \mathbf z) = g(\mathbf x)\kappa_1(\mathbf x, \mathbf z)g(\mathbf z)$也是核函数
4. 软间隔与正则化
	$$\min_{\mathbf w, b}\frac12||\mathbf w||^2 + C\sum_{i = 1}^ml_{0/1}(y_i(\mathbf w^\top\mathbf x_i + b) - 1)$$
	- 0/1损失函数：$l_{0/1}(z) = \begin{cases}1\text{, if } z < 0 \\ 0\text{, otherwise}\end{cases}$
	- 替代损失：
		- hinge损失：$l_{\mathtt{hinge}}(z) = \max(0, 1 - z)$
		- 指数损失：$l_{\exp}(z) = \exp(-z)$
		- 对率损失：$l_{\log}(z) = \log(1 + \exp(-z))$
	continue...
5. 支持向量回归
6. 核方法

[返回](../readme.md)