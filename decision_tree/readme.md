1. [基本流程](training_and_visualizing_a_decision_tree.ipynb)
2. 划分选择
	1. 信息增益
	    - 信息熵：$\mathrm{Ent}(D) = -\sum_{k = 1}^{|\mathcal Y|}p_k\log_2p_k$
	    - 信息增益：$\mathrm{Gain}(D, a) = \mathrm{Ent}(D) - \sum_{v = 1}^V\frac{|D^v|}{|D|}\mathrm{Ent}(D^v) \Rightarrow a^\ast = \argmax_{a \in A}\mathrm{Gain}(D, a)$
	2. 增益率
		$$\mathrm{Gain\_ratio}(D, a) = \frac{\mathrm{Gain}(D, a)}{\mathrm{IV}(a)}, \mathrm{IV}(a) = -\sum_{v = 1}^V\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$
	3. 基尼指数
		$$\mathrm{Gini}(D) = \sum_{k = 1}^{|\mathcal Y|}\sum_{k^\prime \neq k}p_kp_{k^\prime} = 1 - \sum_{k = 1}^{|\mathcal Y|}p_k^2$$
		直观来说，$\mathrm{Gini}(D)$反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。因此，$\mathrm{Gini}(D)$越小，则数据集$D$的纯度越高
		
		属性$a$的基尼指数定义为：$\mathrm{Gini\_index}(D, a) = \sum_{v = 1}^V\frac{|D^v|}{|D|}Gini(D^v) \Rightarrow a^\ast = \argmin_{a \in A}\mathrm{Gini\_index}(D, a)$
4. [剪枝处理](regularization_hyperparameters.ipynb)
5. [连续值处理](regression.ipynb)

[返回](../readme.md)