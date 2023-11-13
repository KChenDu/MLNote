1. 参数范数惩罚：$\tilde J(\mathbf\theta; X, \mathbf y) = J(\mathbf\theta; X, \mathbf y) + \alpha\Omega(\mathbf\theta)$
    1. $L^2$参数正则化
        $$\tilde J(\mathbf w; X, \mathbf y) = \frac\alpha2\mathbf w^\top\mathbf w + J(\mathbf w; X, \mathbf y) \Rightarrow \nabla_{\mathbf w}\tilde J(\mathbf w; X, \mathbf y) = \alpha\mathbf w + \nabla_{\mathbf w}J(\mathbf w; X, \mathbf y) \Rightarrow \mathbf w \leftarrow \mathbf w - \epsilon(\alpha\mathbf w + \nabla_{\mathbf w}J(\mathbf w; X, \mathbf y)) = (1 - \epsilon\alpha)\mathbf w - \epsilon\nabla_{\mathbf w}J(\mathbf w; X, \mathbf y)$$
        $$\mathbf w^* = \argmin_{\mathbf w}J(\mathbf w) \Rightarrow \hat J(\mathbf\theta) = J(\mathbf w^*) + \frac12(\mathbf w - \mathbf w^*)^\top H(\mathbf w - \mathbf w^*) \Rightarrow \nabla_{\mathbf w} J(\mathbf w) = H(\mathbf w - \mathbf w^*)$$
        $$\alpha\tilde{\mathbf w} + H(\tilde{\mathbf w} - \mathbf w^*) = 0 \Rightarrow (H + \alpha I)\tilde{\mathbf w} = H\mathbf w^* \Rightarrow \tilde{\mathbf w} = (H + \alpha I)^{-1}H\mathbf w^*$$
    2. $L^1$参数正则化
4. 数据集增强
8. 提前终止

[返回](readme.md)