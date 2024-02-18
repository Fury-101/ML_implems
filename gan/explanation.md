Based heavily off of [this paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).
We simultaneously train a generator $G$ and a discriminator $D$. The generator attempts to generate fakes images that look like the training data, while the discriminator attempts to find whether an image is fake or not (binary classification). An optimal equilibrium for this would be if $G$ generates fakes that look exactly like the training data and $D$ has an accuracy of $\approx 50$%&ndash;just guessing between fake or real.

$D(x) = $ probability that image $x$ is from the training data.

Let $z$ be a random vector sampled from a standard normal distribution($\mu = 1, \sigma = 0$). 
$G(z)$ maps this random vector to an image by estimating the distribution of the training data, $p_{data}$ to be $p_g$.

From the paper, our full GAN training loss function is:
$ \min\limits_G\max\limits_DV(D, G) = \underbrace{\mathbb{E}_{x\sim p_{data}(x)}[\log D(x)]}_{D \text{ maximizes probability of predicting reals}} + \underbrace{\mathbb{E}_{z\sim p_{z}(z)}[\log (1 - D(G(z)))]}_{G \text{ minimizes probability of D being correct}}$

Note that $\mathbb{E}$ is the expected value operator of a distribution.

The original paper utilizes perceptrons (fully connected layers), while I will use convolutional layers to better reflect the task of image generation, along the lines of this [this paper](https://arxiv.org/pdf/1511.06434.pdf). The discriminator will be a standard CNN with image input and probability output. The generator will be similar but use transposed convolutions.

Transposed convolutions are convolutions which *upsample* the input image, in contrast to a regular convolution which *downsamples* the image.

The generator architecture from the paper:
![Generator architecture](generator_arch.png)

Note the convolutions here are transposed convolutions. 

Here, the Generator initialization takes in two parameters: the size of the input latent vector $z$, and the image map size(default 64) which relates to the size of the features as propagated through the generator.