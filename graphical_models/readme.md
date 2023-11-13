1. 贝叶斯网
    1. 结构
        $$P_B(x_1, x_2, .., x_d) = \prod_{i = 1}^dP_B(x_i | \pi_i) = \prod_{i = 1}^d\theta_{x_i | \pi_i}$$
        ```mermaid
		graph
			subgraph 同父结构
			x11((x1))
			
			x11 --> x3((x3))
			x11 --> x41((x4))
			end
			subgraph V型结构
			x42((x4))
			
			x1((x1)) --> x42
			x2((x2)) --> x42
		    end
		    subgraph 顺序结构
			x((x))
			
			x --> y((y))
			z((z)) --> x
		    end
		```
        在“同父”结构中，给定父节点$x_1$的取值，则$x_3$与$x_4$条件独立。在“顺序”结构中，给定$x$的值，则$y$与$z$条件独立。V型结构亦称“冲撞”结构，给定子节点$x_4$的取值，$x_1$与$x_2$必不独立；奇妙的是，若$x_4$的取值完全未知，则V型结构下$x_1$与$x_2$却是相互独立的
        $$P(x_1, x_2) = \sum_{x_4}P(x_1, x_2, x_4) = \sum_{x_4}P(x_4 | x_1, x_2)P(x_1)P(x_2) = P(x_1)P(x_2)$$
        这样的独立性称为“边际独立性”，记为$x_1\perp\!\!\!\perp x_2$
        
        为了分析有向图中变量间的条件独立性，可使用“有向分离”：
	        - 找出有向图中的所有V型结构，在V型结构的两个父结点之间加上一条无向边
	        - 将所有有向边改为无向边
		由此产生的无向图成为“道德图”，令父结点相连的过程称为“道德化”
		假定道德图中有变量$x$，$y$和变量集合$\mathbf z = {z_i}$，若变量$x$和$y$能在图上被$\mathbf z$分开，即道德图中将变量集合$\mathbf z$去除后，$x$和$y$分属两个连通分支，则称变量$x$和$y$被$\mathbf z$有向分离，$x \perp y | \mathbf z$成立。
	2. 学习
	3. 推断
2. 隐马尔可夫模型
    $$P(x_1, y_1, ..., x_n, y_n) = P(y_1)P(x_1 | y_1)\prod_{i = 2}^nP(y_i | y_{i - 1})P(x_i | y_i)$$
    - 状态转移概率：$a_{ij} = P(y_{t + 1} = s_j | y_t = s_i), 1 \leq i, j \leq N$
    - 输出观测概率：$b_{ij} = P(x_t = o_j | y_t = s_i), 1 \leq i \leq N, 1 \leq i \leq N, i \leq j \leq M$
    - 初始状态概率：$\pi_i = P(y_1 = s_i)$
3. 马尔可夫随机场 
	```mermaid
	graph
		x1((x1))
		x2((x2))
		x3((x3))
		x5((x5))
		x6((x6))
		
		x1 --- x2
		x1 --- x3
		x2 --- x4((x4))
		x2 --- x5
		x2 --- x6
		x3 --- x5
		x5 --- x6
	```
	- 对于$n$个变量$\mathbf x = \{x_1, x_2, ..., x_n\}$，所有团构成的集合为$\mathcal C$，与团$Q \in \mathcal C$对应的变量集合记为$\mathbf x_Q$，则联合概率定义为$P(\mathbf x) = \frac1Z\prod_{Q \in \mathcal C}\psi_Q(\mathbf x_Q) \Rightarrow P(\mathbf x) = \frac1{Z^\ast}\prod_{Q \in \mathcal C^\ast}\psi_Q(\mathbf x_Q)$，其中$\psi_Q$为与团$Q$对应的势函数，用于对团$Q$中的变量关系进行建模，极大团构成的集合为$\mathcal C^\ast$，$Z^\ast = \sum_{\mathbf x}\prod_{Q \in \mathcal C^\ast}\psi_Q(\mathbf x_Q)$为规范化因子
	- 全局马尔可夫性：给定两个变量子集的分离集，则这两个变量子集条件独立
		```mermaid
		graph
		xC((xC))
		
		xA((xA)) --- xC
		xB((xB)) --- xC
		```
		$$P(x_A, x_B, x_C) = \frac1Z\psi_{AC}(x_A, x_C)\psi_{BC}(x_B, x_C) \Rightarrow P(x_A, x_B | x_C) = P(x_A | x_C)P(x_B | x_C)$$
	- 局部马尔可夫性：给定某变量的邻接变量，则该变量条件独立于其他变量。形式化地说，令$V$为图的结点集，$n(v)$为结点$v$在图上的邻接结点，$n^\ast(v) = n(v) \cup \{v\}$，有$\mathbf x_v \perp \mathbf x_{V \backslash n^\ast(v)} | \mathbf x_{n(v)}$
	- 成对马尔可夫性：给定所有其他变量，两个非邻接变量条件独立。形式化地说，令为图的结点集和边集分别为$V$和$E$，对图中的两个结点$u$和$v$，若$\langle u, v\rangle \notin E$，则$\mathbf x_u \perp \mathbf x_v | \mathbf x_{V \backslash \langle u, v\rangle}$
	
	为了满足非负性，指数函数常被用于定义势函数，即$\psi_Q(\mathbf x_Q) = e^{-H_Q(\mathbf x_Q)}$
	
	$H_Q(\mathbf x_Q)$是一个定义在变量$\mathbf x_Q$上的实值函数，常见形式为$H_Q(\mathbf x_Q) = \sum_{u, v \in Q, u \neq v}\alpha_{uv}x_ux_v + \sum_{v \in Q}\beta_vx_v$

[返回](../readme.md)