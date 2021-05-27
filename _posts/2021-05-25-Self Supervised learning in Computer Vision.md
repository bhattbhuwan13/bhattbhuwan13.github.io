---
layout: post
title: Self supervised learning in computer vision
tags: cnn computer-vision self-supervised 
---



Deep learning has proved to be very effective in computer vision problems. Neural network these days can even [surpass human level accuracy in some tasks(check figure 2)](https://arxiv.org/pdf/1706.06969.pdf). Most of these networks use CNN(convolutional neural network) as their basic structure. The performance of these CNNs depends on:  

- Their capability i.e the architecture
- The amount of labelled training data

Today, we have good architectures like VGG, ResNet, DenseNet, GoogLeNet, etc. that can learn complex representations or features in an image. However, to get the best out of these complex architectures we need to train them with a huge amount of labelled data.  For instance, these networks are trained on the **imagenet dataset which consists of 14 million images** belonging to 1000 different classes. But, getting such huge labelled dataset is not feasible for all tasks. So, the question still remains.

## How can we effectively train CNNs with small amount of labelled data?

One solution to this problem is [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning).

> **Transfer learning (TL)** is a research problem in [machine learning](https://en.wikipedia.org/wiki/Machine_learning) (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
>
> -- wikipedia

Using transfer learning we can re-train a model trained to solve one problem, also called pretrained model, to solve another similar problem. For example, we can use a model trained to classify plant leaves to classify plant diseases. But, the problem with this approach is, we may not find a good pretrained model for the task we are trying to perform. Let's say we want do some domain specific or novel task like segmenting specific region from brain CT scan images, then it is hard to find a good(or any)  pre-trained model that we can use for this task. What do we do in such cases? **The answer is, we can leverage self supervised technique to learn useful features from unlabelled data**.



## Self supervised learning

On comparing labelled data with unlabelled data, we will find that, the proportion of unlabelled data is much higher than that of labelled data. 

![data_in_internet](../images/2021-05-25-ssl-cv/data_in_internet.png)

Self supervised learning is a technique that allows us to use the unlabelled data present in the internet or our local devices to train our models to learn features. 

Before discussing self supervised learning, let's define some terms that are crucial to self supervised learning.

- **Pretext Task**:   

  Pretext task is a dummy task on which a networks is trained to learn useful features. The pretext task uses pseudo labels for training.

- **Downstream Task**:

  Downstream task is the actual task that we want to train and evaluate our model on. When we have scarce labelled data, the downstream task will benefit from the features learned while training for the pretext task. The downstream task requires labelled data for training.  

- **Pseudo Label**

  Pseudo label are the automatically generated labels based on data attributes for pretext tasks. 

Self supervised learning is the subset of unsupervised learning. It is the method in which a neural network is trained using pseudo label to solve the pretext task. While solving the pretext task, the network learns features. The network is finally trained on human labelled data to solve the downstream task.



The <authors hypothesize that the> residual mappings are much convenient to optimize than the original unreferenced mapping. If the identity mapping were optimal, then it is easier for the network to simply push the weights corresponding to the residual to zero instead of fitting an identity mapping using nonlinear layers.



**But, in case of a neural networks, is an identity mapping the optimal function?**. The answer is no, identity mapping is not the optimal function. However, experiments show that, in case of residual networks, the layer response is close to zero indicating that the optimal function is very close to the identity mapping and is obtained my making minor adjustments to the identity mapping.  

![Layer response in deep residual networks](/images/2021-03-10-resnet/layer-response.png)



The above figure compares the layer response of  residual network with non-residual counterparts. The bottom figure represents the layers response arragned in descending order. Layer response for residual networks is low and close to zero compared to the plain networks. The response is even lower in case of deeper residual networks. 



Hence, since the optimal function is close to the identity mapping, residual units help to the network converge faster and make the deep network easier to optimize. 



## References

1. [Going Deeper with Convolutions, Szegedy et. al.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)

2. [Deep Residual Learning for Image Recognition, He et. al.](https://arxiv.org/pdf/1512.03385.pdf)
