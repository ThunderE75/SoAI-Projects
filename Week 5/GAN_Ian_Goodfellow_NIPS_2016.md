# Generative Adversarial Networks with Ian Goodfellow - Neural Information Processing Systems Conference - NIPS 2016

- https://arxiv.org/pdf/1701.00160
- https://learn.microsoft.com/en-us/shows/neural-information-processing-systems-conference-nips-2016/generative-adversarial-networks?wt.mc_id=studentamb_291695

### Major Questions 
- Why study Generative Modeling ?
- How do they work? how GAN is compared to others
- How do GANs work ?

---

![Next-video-frame-prediction](<../Resorces/Images/Week 5/Next-video-frame-prediction.png>)

- Ground Truth → The expected final output of the rotating head
- MSE (Mean Squared Error) → Since the head can move in many direction, this traditional method takes in all the possibility and averages them together.
- Adversarial → This model produces a better image output

---

## Single Image Super-Resolution

![Single Image Super-Resolution](<../Resorces/Images/Week 5/Single Image Super-Resolution.png>)

1. Here we take the original image, & down sample it to half its resolution (not shown in the example)
2. Now we try reconstruct the images using various methods 
    1. Bicubic
    2. SRResNet
    3. SRGAN (Super Resolution GAN)

---

![alt text](<../Resorces/Images/Week 5/tree.png>)

## Fully visible belief networks (FVBNs )

These models that use the chain rule of probability to decompose a probability distribution over an $n$-dimensional vector $x$ into a product of one-dimensional probability distributions

$$
p_{model}(x) = \overset{n}{\underset{i=1}{\Pi}}~~Pmodel(x_i | x_1, ..., x_i)
$$

They are one of the three most popular approaches to generative modeling, alongside **GANs** and variational autoencoders (VAE).

### Disadvantage

- $O(n)$ sample generation cost
- generation not controlled by latent code

### WaveNet
![Wavenet](<../Resorces/Images/Week 5/Wavenet.png>)
(by DeepMind)
- Amazing Quality
- Slow generation time 
(2 min to synthesize 1 sec of audio)

---

# Generative Adversarial Networks (GANs)

> 🔵  Made by Dr. Ian Goodfellow & colleagues in 2014

- It consist of two neural networks that compete against each other:
    - Generator
    - Discriminator
- This adversarial process helps the generator to improve over time, leading to the creation of realistic data such as images, videos, and text.
- No Markov chain needed

## The GAN framework

- GANs are a game between Generator & Discriminator
    - The generator creates samples that are intended to come from the same distribution as the training data.
    - The discriminator examines samples to determine whether they are real or fake.
- The discriminator learns using traditional supervised learning techniques, dividing inputs into two classes (real or fake).
- The generator is trained to fool the discriminator.

---

> 🔥 Never thought i would experience the drama → Timestamp [01:02:57]