1. 间隔与支持向量
    $$\mathbf \omega^\top\mathbf x + b = 0$$
    样本空间中任意点$\mathbf x$到超平面$(\mathbf w, b)$的的距离可写为$r = \frac{|\mathbf \omega^\top\mathbf x + b|}{||\mathbf \omega||}\Rightarrow \gamma = \frac{2}{||\omega||}$
    
    两个异类支持向量到超平面的距离之和为$\gamma = \frac{2}{||\omega||}$
    $$\max_{\omega, b}\frac{2}{||\omega||}\text{ s.t. }y_i(\mathbf \omega^\top\mathbf x_i + b) \ge 1 \Leftrightarrow \min_{\omega, b}\frac{||\omega||}{2}\text{ s.t. }y_i(\mathbf \omega^\top\mathbf x_i + b) \ge 1$$
2. 对偶问题
	$$L(\mathbf\omega, b, \mathbf\alpha) = \frac12||\mathbf \omega||^2 + \sum_{i = 1}^m\alpha_i(1 - y_i(\mathbf\omega^\top\mathbf x_i + b)) \Rightarrow
	\begin{cases}
		\mathbf\omega = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i \\
		0 = \sum_{i = 1}^m\alpha_iy_i
	\end{cases}
	\Rightarrow \max_{\mathbf\alpha}\sum_{i = 1}^m\alpha_i - \frac12\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_j\mathbf x_i^\top\mathbf x_j\text{ s.t. }\sum_{i = 1}^ma_iy_i = 0\text{, }\alpha_i \ge 0\text{, }i = 1, 2, ..., m \Rightarrow f(\mathbf x) = \mathbf\omega^\top\mathbf x + b = \sum_{i = 1}^m\alpha_iy_i\mathbf x_i^\top\mathbf x + b$$
	- KTT条件：$\begin{cases}\alpha_i \ge 0; \\ y_if(\mathbf x_i) - 1 \ge 0 \\ \alpha_i(y_if(\mathbf x_i) - 1) = 0\end{cases}$
3. 核函数
4. 软间隔与正则化
	$$\min_{\mathbf\omega, b}\frac12||\mathbf\omega||^2 + C\sum_{i = 1}^ml_{0/1}(y_i(\mathbf\omega^\top\mathbf x_i + b) - 1)$$
	- 0/1损失函数：$l_{0/1}(z) = \begin{cases}1\text{, if } z < 0 \\ 0\text{, otherwise}\end{cases}$
	- 替代损失：
		- hinge损失：$l_{\mathtt{hinge}}(z) = \max(0, 1 - z)$
		- 指数损失：$l_{\exp}(z) = \exp(-z)$
		- 对率损失：$l_{\log}(z) = \log(1 + \exp(-z))$

[返回](../readme.md)