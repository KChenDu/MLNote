10. 有向生成网络

    3. [变分自编码器](variational_autoencoder.ipynb)
        1. **The neural net perspective**: a variational autoencoder consists of an encoder, a decoder, and a loss function
            - *encoder* ($q_\theta(z | x)$): a neural network. Its input is a datapoint $x$, its output is a hidden representation $z$, and it has weights and biases $\theta$
            - *decoder* ($p_\phi(x | z)$): another neural net. Its input is the representation $z$, it outputs the parameters to the probability distribution of the data, and has weights and biases $\phi$

            **How much information is lost?**
            We measure this using the reconstruction log-likelihood $\log p_{\phi}(x | z)$ whose units are nats.

            - *loss function*: the negative log-likelihood with a regularizer. Because there are no global representations that are shared by all datapoints, we can decompose the loss function into only terms that depend on a single datapoint $l_i$. The total loss is then $\sum_{i = 1}^Nl_i$ for $N$ total datapoints.
                $$l_i(\theta, \phi) = -\mathbb E_{z \sim q_\theta(z | x_i)}[\log p_\phi(x_i | z)] + \mathrm{KL}(q_\theta(z | x_i) \| p(z))$$
                The first term is the reconstruction loss. The second term is a regularizer that we throw in.
            In the variational autoencoder, $p$ is specified as a standard Normal distribution with mean zero and variance one, or $p(z) = \mathrm{Normal}(0, 1)$.

            We train the variational autoencoder using gradient descent to optimize the loss with respect to the parameters of the encoder and decoder $\theta$ and $\phi$. For stochastic gradient descent with step size $\rho$, the encoder parameters are updated using $\theta \leftarrow \theta - \rho\frac{\partial l}{\partial\theta}$ and the decoder is updated similarly.
        2. **The probability model perspective**: a variational autoencoder contains a specific probability model of data $x$ and latent variables $z$
            We can write the joint probability of the model as $p(x, z) = p(x | z)p(z)$

            For each datapoint $i$:
            - Draw latent variables $z_i \sim p(z)$
            - Draw datapoint $x_i \sim p(x | z)$
            $$p(z | x) = \frac{p(x | z)p(z)}{p(x)}$$
            Variational inference approximates the posterior with a family of distributions $q_\lambda(z | x)$. The variational parameter $\lambda$ indexes the family of distributions.

            **How can we know how well our variational posterior $q(z | x)$ approximates the true posterior $p(z|x)$?**
            We can use the Kullback-Leibler divergence, which measures the information lost when using $q$ to approximate $p$ (in units of nats): $\mathrm{KL}(q_\lambda(z | x) \| p(z | x)) = \mathbb E_q[\log q(z | x)] - \mathbb E_q[\log p(x, z)] + \log p(x) \Rightarrow q_\lambda^*(z  |x) = \argmin_\lambda \mathrm{KL}(q_\lambda(z | x) \| p(z | x))$
            $$\mathrm{ELBO}(\lambda) = \mathbb E_q[\log p(x, z)] - \mathbb E_q[\log q_\lambda(z | x)] \Rightarrow \log p(x) = \mathrm{ELBO}(\lambda) + \mathrm{KL}(q_\lambda(z | x) \| p(z | x))$$
            The ELBO for a single datapoint in the variational autoencoder is: $\mathrm{ELBO}_i(\lambda) = \mathbb E_{q_\lambda(z | x_i)}[\log p(x_i | z)] - \mathrm{KL}(q_\lambda(z | x_i) \| p(z))$
        3. **Connection**
            We can write the ELBO and include the inference and generative network parameters as: $\mathrm{ELBO}_i(\theta, \phi) = \mathbb E_{q_\theta(z | x_i)}[\log p_\phi(x_i | z)] - \mathrm{KL}(q_\theta(z | x_i) \| p(z)) \Rightarrow \mathrm{ELBO}_i(\theta, \phi) = -l_i(\theta, \phi)$

[返回](readme.md)