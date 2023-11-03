1. 基本流程
    ![DT](decision_tree.png "决策树学习基本算法")
    显然，决策树的生成是一个递归过程。在决策树基本算法中，有三种情形会导致递归返回: 
    1. 当前结点包含的样本全属于同一类别，无需划分
    2. 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分
    3. 当前结点包含的样本集合为空，不能划分

    在第2种情形下，我们把当前结点标记为叶结点，井将其类别设定为该结点所含样本最多的类别；在第3种情形下，同样把当前结点标记为叶结点，且将其类别设定为其父结点所含样本最多的类别。注意这两种情形的处理实质不同：情形2是在利用当前结点的后验分布，而情形3则是把父结点的样本分布作为当前结点的先捡分布
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
	4. 剪枝处理



[返回](../readme.md)