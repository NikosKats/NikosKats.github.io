# This is a short summary for Neural Networks and Deep Learning course by Andrew Ng.

## Week 1 - Introduction to Deep Learning

A **Neuron** is inspired by the biological neuron and is the basic building block of neural networks. Are responsible for computing and transmitting information through the network.

A neuron consists of a set of inputs, called dendrites, and a single output, called an axon. The inputs of a neuron are connected to the outputs of other neurons, forming a network of connections. Each connection is associated with a weight, which determines the strength of the connection.

![alt text](/img/postimages/neuralNetwork.jpg)

Neural Networks are inspired by the human brain and they are comprised of node layers or neurons.

Each node or artificial neuron is connected to another neuron and has an associated weight and threshold. If the output of any individual neuron is above the specified threshold value, the neuron is activated, sending data to the next layer of the network. Otherwise no data is passed along to the next layer of the network.

> Deep Learning is rising in recent times because of the amount of data available as well as the computational power provided. The other important factor is the development of even better deep learning algorithms and the invention of new more capable ones by the research community.

---

## Week 2 – Logistic Regression as a Neural Network (1 of 2)

In a Binary Classification problem, the result is a discrete value output - account hacked (1) or not hacked (0) - is a cat image (1) or a non-cat image (0).

**Logistic Regression** is an algorithm for Binary Classification, it estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. 𝐺𝑖𝑣𝑒𝑛 𝑥, 𝑦̂=𝑃(𝑦=1|𝑥), where 0 ≤𝑦̂≤1

![alt text](/img/postimages/logisticRegression.png)
