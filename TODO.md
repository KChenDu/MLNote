Why does regularization reduce Var of model?
$$\hat w = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y = \Phi(\Phi\Phi^\top + \lambda I)^{-1}y$$
$$(\Phi\Phi^\top + \lambda I)^{-1} = \alpha$$

$$f(x) = \hat w^\top \phi(x) = \alpha^\top\Phi\Phi(x) = \alpha^\top$$
$k(x, z) = \Phi(x)\Phi(z)$ -> reduces complexity

k -> kernel

$$\alpha = (k + \lambda I)^{-1}$$
$$$$