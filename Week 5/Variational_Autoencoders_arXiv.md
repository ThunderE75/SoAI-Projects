# Variational Autoencoders (VAE) - arXiv

> 💠 by Navneet Prakash Dubey (MCA)

- https://youtu.be/9zKuYvjFFS8
- https://arxiv.org/abs/1312.6114
- https://arxiv.org/abs/1606.05579
- https://arxiv.org/abs/1707.08475

# Aim of VAE (**V**ariational **A**uto**E**ncoders)

- Takes in any data (image or vector ) anything with a high dimensionality
- Run it through a NN
- & try to compress the data into a smaller representation

---

## Components of Traditional Auto Encoder

- Encoder
- Bottleneck
- Decoder

> The layers can either be fully connected layers or Convolutional layers

> 🔵 Reconstruction can be calculated by comparing reconstructed & original data.

![Autoencoder Architecture](<../Resorces/Images/Week 5/autoencoder_architecture.png>)

---

Image Segmentation also works like this, example is `SegNet` 

![SegNet](<../Resorces/Images/Week 5/Segnet.png>)

---

### Denoising Auto encoder

![Denoising Auto encoder](<../Resorces/Images/Week 5/architecture-of-denoise.jpg>)

1. We take the MNIST dataset.
2. we add artificial noise
3. we pass it through the VAE
4. we try to construct the original image (not the image with artificial noise)

> With enough training, we can make this a Denoising Auto encoder

---

### Neural Inpainting

![Neural Inpainting](<../Resorces/Images/Week 5/Neural Inpainting.png>)

1. We take an image & we destroy a certain part of it
by adding a opaque box over it.
2. We then pass it through the network
3. we attempt to reconstruct the original image

> Can be used for watermark removal, removing object from photos & videos

---

---

# Variational AutoEncoder

> Here, instead of mapping an input to a fixed vector, we instead map it to a distribution

> Its different from Traditional Auto Encoder because instead of a bottleneck vector, we replace it with 2 vectors - One Representation the mean of distribution & the other representing standard deviation


![Variational AutoEncoder Architecture](<../Resorces/Images/Week 5/Variational AutoEncoder Architecture.png>)
![Reconstruction Loss](<../Resorces/Images/Week 5/Reconstruction Loss.png>)


## Reparameterization Trick

$$
z = \mu + \sigma \odot \epsilon \\ where~\epsilon~\text{\textasciitilde}~Normal(0,1)
$$

![Reparameterization Trick](<../Resorces/Images/Week 5/Reparameterization Trick.png>)

## Disentangled Variational Autoencoders

- Disentangled Variational Autoencoders (DVAEs) are a type of VAE that aim to learn a latent representation where each dimension corresponds to a distinct, interpretable factor of variation in the data.
- To achieve disentanglement, an additional hyperparameter (β) is added to the loss function that controls the strength of the KL divergence term.
- By increasing β, the DVAE is forced to use only a few latent dimensions to encode the input, resulting in a more disentangled representation.
- Disentangled representations can be useful for tasks like reinforcement learning, where the agent can learn useful behaviors on the compressed latent space.

![Disentangled Variational Autoencoders](<../Resorces/Images/Week 5/Disentangled Variational Autoencoders.png>)

## Applications and Tradeoffs

- VAEs and DVAEs have been applied to various domains, such as reinforcement learning, where the compressed latent representation can be used as input to the agent.
- There is a tradeoff when training VAEs and DVAEs:
    - If the latent space is not disentangled enough, the network may overfit to the training data and not generalize well.
    - If the latent space is too disentangled, the network may lose important high-dimensional details, which can hurt performance in certain applications.

| Comparison | Disentangled VAE | Normal VAE |
| --- | --- | --- |
| Latent Space Interpretation | Each latent dimension corresponds to a distinct, interpretable factor of variation | Latent dimensions are not easily interpretable |
| Generalization to Unseen Data | Better generalization due to the disentangled representation | May not generalize as well as the disentangled VAE |
| Reconstruction Quality | May lose some high-dimensional details due to the strong disentanglement constraint | Can preserve more high-dimensional details |