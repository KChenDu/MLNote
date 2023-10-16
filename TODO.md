COMP78

    Why does regularization reduce Var of model?
    $$\hat w = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y = \Phi(\Phi\Phi^\top + \lambda I)^{-1}y$$
    $$(\Phi\Phi^\top + \lambda I)^{-1} = \alpha$$

    $$f(x) = \hat w^\top \phi(x) = \alpha^\top\Phi\Phi(x) = \alpha^\top$$
    $k(x, z) = \Phi(x)\Phi(z)$ -> reduces complexity

    k -> kernel

    $$\alpha = (k + \lambda I)^{-1}$$


COMP80

    (ex, ey): explosion loc
    (xi, yi): detector i
    measure: 1 / (di ^ 2 + 0.1)
    di^2 = (xi - ex)^2 + (yi - ey)^2
    P(vi | di) = Normal(vi; sigma)
    P((ex, ey), v1, ..., vn)

    ```mermaid
    (ex, ey) --> v1
            --> v2
            ...
    ```
    P((ex, ey) | v1, ..., vn) = ?