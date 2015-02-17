---
layout: post
title: Introduction to Conditional Random Fields
---

<script type="text/javascript"
src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# THIS POST IS IN PROGRESS. IT'S ONLY ONLINE SO I CAN TEST OUT JEKYLL W/ J/MATHJAX

It's been a long time since I've written any blog posts. I'm going to start a series talking about structured prediction. The focus will mostly be on time series models targeted at vision applications -- however many of the ideas will be much more far reaching. In the future I will touch on some specific models (e.g. Skip Chain CRF, Semi Markov CRF, Latent CRF), methods of learning (e.g. for solving Structural SVMs), and various other ideas relating to structured prediction. In addition, at some point I will be releasing my structured prediction library *struct.jl* that I have been writing in Julia. 

To start I will give an overview of Conditional Random Fields (CRFs). CRFs are incredibly general and have widespread use in vision, robotics, and beyond. There are three necessary components for understanding and implementing CRFs: 

 - **Model:**  A definition of what information we know (e.g. data), what information we want to infer (e.g. labels/states), and how they all connect. This is often defined as a graph structure.
 - **Inference:** A method for determining the states given a set of data.
 - **Learning:** A method for learning how much to weigh each of the connections in our graph.

<!-- To start I will give an overview of Conditional Random Fields (CRFs). CRFs are incredibly general and have widespread use in vision, robotics, and beyond. There are three necessary components: first, we must define a **model**. This is a definition of what information we know (e.g. data), what information we want to infer (e.g. labels/states), and how it all connects. This is often defined as a graph structure where nodes represent data/states and edges represent the connections between the data. Second, we need a way of performing **inference**; how can we determine which states most likely compute the best labeling given a set of data. Third, we must **learn** how much to weigh each of the connections in our graph. -->

We will use two motivating examples throughout: a Linear Chain CRF and an (irregular) Graph CRF as shown below. In the diagrams Colored circles represent labels (e.g. of class red and blue) and the white squares represent data points. Linear chain models are very common for working with time series data. The key assumption is that the label at each timestep is dependent only on the current data term and the previous label in time. Graph CRFs are commonly used for problems like Semantic Segmentation where you may have a set of segments in an image and want to infer what object each segment refers to.

![Linear Chain CRF](https://lh6.googleusercontent.com/-K2KUQYbOAi0/VOJmNDkrwqI/AAAAAAAAFJQ/4gPkC0Ag6co/s400/Screenshot+2015-02-16+16.49.37.png "Linear Chain CRF")

<center> Linear Chain Conditional Random Field </center>

![Graph CRF](https://lh6.googleusercontent.com/-ylh-Ok2PQ_A/VOJmEdpihWI/AAAAAAAAFJE/XEroJ5harss/s400/Screenshot+2015-02-16+16.47.33.png "Graph CRF")

 <center>Graph Conditional Random Field</center>


The focus of this post will be on the model definition but I will also discuss learning and inference. In some future posts I will very concretely describe some example models and applications that I have developed.

##Model
A CRF can be represented by a graph. For now we assume that there are two types of nodes: labels which are denoted as $$Y_i$$ and data which is denoted as $$X_i$$ for some index $$i$$. Note that we have access to both the labels and the data during training. At test time our goal is to infer the best set of labels given our data. 

Let's start by modeling individual connections between nodes. In the linear chain case, we want to model the connection between the label at any timestep and the associated data at that timestep as $$\phi(X_t, Y_t)$$. In addition we want to model how the labels transition over time as $$\psi(Y_t, Y_{t-1})$$. The data-to-label term is called a *unary potential* and the label-to-label term is called a *pairwise potential*. These typically have vector-valued inputs of size $$n$$ and outputs of size $$m$$ ($$\psi: \mathcal{R}^n \rightarrow \mathcal{R}^m$$). Note that the terms *potential,* *cost function,* and *factor* are synonymous here.

In the Graph CRF we include a connection between multiple labels $$\gamma(Y_{i:j})$$. This is called a higher-order potential. In another example, we could model a triplet of labels $$Y_{t-2:2}$$ from time $$t-2$$ to $$t$$ in a linear chain CRF. Typically inference and learning with higher order potentials is more complex computationally and memory-wise than with just unary and pairwise terms.  

In the end we want to predict the best scoring labels for a whole network as opposed to the score from each individual potential. This means we need to define an energy function that takes in all of the potentials. For clarity let's first define an abstract potential $$\Psi_i$$ of arbitrary potential type with index $$i$$. We define the probability distribution for each potential using the exponential distribution $$P(Y_i \vert X_i) \propto \exp(w^T\Psi_i(X_i, Y_i))$$ where $$w$$ is a weight vector (classifier) that is learned. $$w$$ defines how important each of the features is for a given class.

For the whole CRF, we model the conditional distribution of the labels $$Y$$ given the data $$X$$ and weight vector (classifier) $$w$$ as a Gibbs distribution:
<center>$$P(Y \vert X) = \frac{1}{Z} \prod_{i=1}^M \exp(w^T \Psi_i(X, Y))$$</center>
This expression is extremely general and implies that the probability of the labels given the data is proportional to the product of the probabilities of each potential. Here we assume there are $$M$$ potentials. In the linear chain case we can define this to be the number of timesteps where $$\Psi_i$$ is now a combination of the unary and pairwise potentials at time $$t$$:  $$\Psi_i(X, Y) = [\phi(X_t, Y_t), \psi(Y_t, Y_{t-1})]$$.

The term $$Z$$ in the previous expression is the normalization term and represents every possible configuration of the labels. For inferring the most likely sequence $$Y_{1:T}$$, and under certain regimes of learning, we only care about inferring the most likely output $$\hat{Y}$$. In these cases we never need to explicitly compute $$Z$$. Thus we can write $$P(Y \vert X) \propto \prod_{i=1}^M \exp(w^T \Psi_i(X, Y))$$.

Note that we for clarity we have overloaded the weight vector (classifier) $$w$$. There is a separate vector $$w_c$$ that corresponds to each class. Moreover, each In our notation we use a single variable $$w$$ due to the 

**To recap:**
 - **Unary** ($$\phi(X_i, Y_i)$$): the cost of some data $$X_i$$ paired with a specific label $$Y_i$$.
 - **Pairwise** ($$\psi(Y_i, Y_j)$$): the cost of two labels $$Y_i$$ and $$Y_j$$ being connected.
 - **Higher-order** ($$\gamma(Y_{i:j}, X_{i:j})$$): the cost of a group of nodes.

 
## Learning
In order to perform inference in a CRF we need the parameter vectors $$w$$ that weigh each the potentials. There are two categories of learning algorithms: probabilistic and max-margin. **Probabilistic** methods are based on maximizing the conditional likelihood of the model. This is the way that the parameters of a CRF were traditionally learned. More recently, the problem has been formulated as a convex optimization problem using the Structural Support Vector Machine (SSVM) objective (a.k.a. the **max-margin** approach). For many applications, SSVM methods tend to be faster and achieve superior accuracy than probabilistic methods. In the future I will write about some of these methods. It took me a while to fully understand the SSVM methods so I think it would be useful to write about them.


<!--
Probabilistic 
A common method is to Maximum Likelihood
$$w = \underset{w}{\arg\max{}} \prod_{i=1}^N P(Y^{(i)} \vert X^{(i)}; w)$$
$$w^t = w^{t-1} + \alpha p $$

Structural Support Vector Machine
The objective function is posed as follows:
--> 

There are three methods for optimization that I think colleagues should be familiar with in regards to max-margin training: subgradient descent, cutting planes, and Block Coordinate Frank Wolfe (BCFW). The subgradient method is the simplest to derive and is also the basis for more advanced methods like the Convex Concave Procedure (CCCP) for learning latent structural models. Cutting planes is commonly used in vision papers and is easy to understand intuitively. BCFW is more complicated but in my experience (and from comparisons I've seen) is faster and achieves higher accuracy than the other methods. 

##Inference
The method in which we perform inference depends on the model definition. In abstract, we want to find the best labels $$\hat{Y}$$ given some data $$X$$.
<center>
$$\hat{Y} = \underset{Y}{\arg\max{}} P(Y \vert X)$$
</center>

At the broadest level there are two categories of inference: exact and approximate. Exact inference methods deterministically compute the best set of labels. Unfortunately exact inference is only possible in acyclic graphs like chains and trees. Approximate inference methods typically rely on some sort of sampling (either on the structure of the graph or on individual node values) and may result in different labelings between different trials or with a different number of samples. 

In addition there are two separate subproblems. We can either try to compute the marginal distribution for each node (e.g. sum-product) or simply the most likely label at each node (e.g. max-product). Here we will focus on finding the marginal distribution.

The most important method to be familiar with is Belief Propagation (a.k.a. message passing). This is the basis for exact inference in trees and is also used in approximate methods such as Loopy BP and Tree Reweighted Message Passing (TRW). Note that there is a vast literature on different inference methods for working with different types of data and model definitions. In BP, the potentials for each label node are collected into beliefs which are passed between nodes recursively. In order to do this algorithm justice I will save its explanation for another time.

<!--
 $$b_t(i)$$ where $$t$$ is the label index, $$i$$ is the current class label, and $$\mathcal{N}(t)$$ is the set of $$t$$'s neighbors:

<center>$$b_t(i) = \sum_{j\in\mathcal{Y}} \prod_{f\in\mathcal{N}(t)} \exp(w^T \Psi_f(X, Y))$$</center>
The sum in this expression is used to marginalize over possible labelings of the neighboring nodes. 
-->

<!--For CRFs, the key problem for inference is to find the best set of labels, $$Y_{1:T}$$, given a set of data, $$X_{1:T}$$ from time 1 to $$T$$. Typically in the linear chain case we assume that the label at some timestep, $$Y_t$$, is dependent only on the data at that timestep and the previous label $$Y_{t-1}$$. -->

##Where else can I learn about CRFs?
For a more in-depth introduction from a machine learning perspective see Sutton and McCallum's [An Introduction to Conditional Random Fields](http://www.nowpublishers.com/articles/foundations-and-trends-in-machine-learning/MAL-013) . For a more recent vision-oriented perspective (including discussion of max-margin based methods) see Nowozin and Lampert's [Structured Learning and Prediction in Computer Vision](http://www.nowpublishers.com/article/Details/CGV-033). PDFs of both of these can be found easily.

For a good implementation I highly suggest [PyStruct](https://pystruct.github.io/). It includes a set of efficient learning techniques paired with wrappers to inference techniques from other libraries (e.g. OpenGM). The primary limitation I've found using pyStruct is that it is hard to prototype new models using it. This is less of a limitation with Andreas' library and has more to do with limitations of Python in general. As such, I am developing a library, Struct.jl, written in Julia which is fast and is easy to modify due to the software design I have chosen. More on this soon.
