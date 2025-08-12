# A Primer on CLIP and Zero-Shot Classification

This document provides a primer on the CLIP architecture and its application in Zero-Shot Learning.

## The Challenge of Traditional Image Classification

A traditional classification problem aims to teach a model, through labeled examples, to which class an image belongs. For instance, in a cat/dog classifier, the model learns that class '0' refers to 'dog' and class '1' refers to 'cat'. Through its internal adjustments during training, it understands which pixel patterns correspond to a dog or a cat. It uses this learned reference to infer the class of new, incoming images.


![CatDogClassifier](/docs/images/cat_dog_classifier.jpg)

However, modern models based on neural networks face a significant challenge: It requires a massive quantity of labeled data to gain a minimum efficiency in its performance. Due that it is require a intensive labor based and a extensive costly process to generate/labelling new data, and often this process is slowly and ineficcient for new problems. For example, imagine if we want to add a new class to our cat/dog classifier, for example a duck. For our classifier understand what a duck is we will need to labelling and rerun the training from scratch to him understand this new class. This has high computational cost and extensive scalability issues.


## The CLIP architecture: A Dual-Encoder Approach

Due that computational and scalability problem, modern approaches tries to use some tecniques that minimizes/removes the need of new data to learn new classes, for example few-learning-shot and zero-shot-learning. The goal of those tecniques is to avoid the necessity of re-run the entire training pipeline and the necesity of labelling new classes.

A new approach is CLIP - Constrative Language-Image Pre-Trained. This tecnique uses a dual-encoder parallel architecture based in a Image encoder and a text encoder. This architecture marks a breakthrough from traditional encoder-decoder models. In CLIP's dual-encoder architecture, the Image Encoder and the Text Encoder operate independently and in parallel. Their goal is not to translate information to each other, but to map their respective inputs (images and text) into a shared embedding space. This means a picture of a dog and the text 'a photo of a dog' are projected to nearby points in this common space, allowing for direct comparison.

![clip_architecture](/docs/images/clip_architecture.svg)

An image encoder (e.g,. Vision Transformer - ViT) is used pair to a Text encoder(e.g,. Masked Language Model, like Transformer). Both encoder is processed separately.


This dual-encoder process is what allows for the creation of multimodal embeddings, where concepts from both text and images coexist in the same vector space.


## Training Methodology: Contrastive Learning


CLIP architecture was trained based on Contrastive learning methodology. This kind of learning has the following idea:

![Constrative Learning](/docs/images/constrative_learning.jpeg)

Beyond an text/image pair it is create positive examples and negative examples based in a anchor. Through this kind of example we adjust the embedding weights so during the cosine similiarity the positive examples become as close as possible to 1 and the negative examples as nearly as possible to -1. For example, if we want to apply this tecnique for a dog classifier, we say as positive example "This is a dog", to an anchor-dog based. And for a negative example we say "This is not a dog" or "This is a owl". Here the concept of Prompt Engineering is fundamental. Through all of this the goal is to optimize the cosine similarity.

![Cosine Similiarity](/docs/images/cosine_similiarity.jpeg)


## The Emergent Capability: Zero Shot Classification


So, based on discussed problems of modern image classification and this new architecture, how it propose to solve those problems, and how it does?

To start, this kind of architecture enable the learning through zero-shot, this deals with the necessity of retrain an entire model just to add one unseen class. This approach saves significant time, effort, and resources. Thinking about the propousal, this is done by semantic search. Let's get back to dog/cat classifier. If we want to add a duck class we just have to add a description about it. For example "It swim, fly, walk" - Of course, for better results we must adjust the best using prompt engineering techiques.

Plus this entire architecture was trained based in million web images, so it possesses a vast contextual understanding of the world.


## Synthesis & Conclusion


In essence, CLIP is a dual-encoder architecture trained via contrastive learning to align image and text embeddings into a shared space, which in turn enables powerful Zero-Shot classification capabilities.