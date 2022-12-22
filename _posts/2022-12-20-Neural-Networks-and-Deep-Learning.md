 <style>
    {
        box-sizing: border-box;
    }
    /* Set additional styling options for the columns*/
    .column {
    float: left;
    width: 50%;
    }

    .row:after {
    content: "";
    display: table;
    clear: both;
    }

    .center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 70%;
    }
</style>

# This is a short summary of Neural Networks and Deep Learning course by Andrew Ng.

## Introduction to Deep Learning

A **Neuron** is inspired by the biological neuron and is the basic building block of neural networks. Are responsible for computing and transmitting information through the network.

A neuron consists of a set of inputs, called dendrites, and a single output, called an axon. The inputs of a neuron are connected to the outputs of other neurons, forming a network of connections. Each connection is associated with a weight, which determines the strength of the connection.

<figure class="center">
    <img src="/img/postimages/neuralNetwork.jpg" alt="Shallow Neural Network" width="100%" height="350px" >
    <figcaption><pre>A shallow neural network consists of an input, 
    a hidden and an output layer.</pre></figcaption>
</figure>

**Neural Networks** are inspired by the human brain and they are comprised of node layers or neurons.

Each node or artificial neuron is connected to another neuron and has an associated weight and threshold. If the output of any individual neuron is above the specified threshold value, the neuron is activated, sending data to the next layer of the network. Otherwise no data is passed along to the next layer of the network.

> Deep Learning is rising in recent times because of the amount of data available as well as the computational power provided. The other important factor is the development of even better deep learning algorithms and the invention of new more capable ones by the research community. [^1]

---

## Logistic Regression as a Neural Network

In a **Binary Classification** problem, the result is a discrete value output - account hacked (1) or not hacked (0) - is a cat image (1) or a non-cat image (0).

**Logistic Regression** is an algorithm for Binary Classification, it estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. `𝐺𝑖𝑣𝑒𝑛 𝑥, 𝑦̂=𝑃(𝑦=1 | 𝑥), where 0 ≤𝑦̂≤1`

<figure class="center">
    <img src="/img/postimages/logisticRegression.png" alt="Shallow Neural Network" width="100%" height="350px" >
    <figcaption><pre>A shallow neural network consists of an input, 
    a hidden and an output layer.</pre></figcaption>
</figure>

The **loss function** measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)). In other words, the loss function computes the error for a single training example.

The **cost function** is the average of the loss function of the entire training set. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.

A **Computation Graph** is a visual representation of the operations performed by a computer program, especially a machine learning or deep learning model.

**Vectorization** is the process of converting a set of scalar values, or a sequence of operations, into a vector or a matrix.

In Python, **broadcasting** is a mechanism that allows arrays of different shapes to be used together in mathematical operations.

**Normalization** is the process of scaling a set of values to have a mean of 0 and a standard deviation of 1.

<div class="row">
        <div class="column" style="background-color:#ffffff;">
            <figure>
                <img src="/img/postimages/shallowNN.png"
                    alt="Shallow Neural Network">
                <figcaption><pre></pre></figcaption>
            </figure>
            <p>A shallow neural network consists of an input, 
                a hidden and an output layer.</p>
        </div>
        <div class="column" style="background-color:#ffffff;">
            <figure>
            <img src="/img/postimages/activationFunctions.png"
                alt="Activation Functions">
            <figcaption><pre></pre></figcaption>
        </figure>
        <p>Activation functions are mathematical equations 
            that determine the output of a neuron in a neural network.</p>
        </div>
</div>

---

<div class="row">
        <div class="column" style="background-color:#ffffff;">
            <figure>
                <img src="/img/postimages/gradientDescent.png"
                    alt="Gradient Descent">
                <figcaption><pre></pre></figcaption>
            </figure>
            <p>Gradient Descent is an optimization algorithm. The algorithm works by iteratively updating the values of the parameters in the direction that reduces the loss. 
            </p>
        </div>
        <div class="column" style="background-color:#ffffff;">
            <figure>
            <img src="/img/postimages/fprop_bprop.png"
                alt="Forward and Backward propagation">
            <figcaption><pre></pre></figcaption>
        </figure>
            <p> 
            In Forward propagation the input data is processed through the network layers to generate output.
            Backpropagation’s goal  is to adjust the weights of the connections in such a way as to minimize the difference between the predicted output of the network and the target output.
            A neural network in contrast with logistic regression that initializes its parameters with zero, has to randomly initialize its parameters in order to work.
            </p>
        </div>
</div>

---

<figure class="center">
    <img src="/img/postimages/gd.png" alt="Gradient Descent" width="100%" height="350px" >
    <figcaption><pre></pre></figcaption>
</figure>

**Gradient Descent** is an optimization algorithm that is used to find the values of the weights and biases that minimize the loss function in a machine learning model. It involves adjusting the values of the weights and biases in small increments based on the gradient of the loss function until the loss is minimized or the model has reached a satisfactory level of performance.

Gradient descent is an optimization algorithm that is used to find the values of parameters (such as weights and biases) that minimize a loss function. It is a key part of the training process for many machine learning models, including neural networks.

In gradient descent, the model makes predictions based on a set of input data and compares the predictions to the true values (also known as the labels). The difference between the predictions and the true values is the loss, which is a measure of how well the model is performing. The goal of gradient descent is to find the values of the model's parameters that minimize the loss.

Weights and biases are model parameters that are used to make predictions. In a neural network, weights are the values that are multiplied by the input data and summed together to produce the input for the activation function. Biases are added to the weighted sum of the inputs and can help the model make predictions that are shifted away from 0.

During the training process, the model uses gradient descent to adjust the values of the weights and biases in order to minimize the loss. The model starts with initial values for the weights and biases and then adjusts these values in small increments based on the gradient of the loss function. The gradient is a vector of partial derivatives that points in the direction of the greatest increase in the loss function. The model moves in the opposite direction of the gradient, which helps to reduce the loss.

The process of adjusting the weights and biases in small increments based on the gradient of the loss function is repeated until the loss is minimized or the model has reached a satisfactory level of performance. The values of the weights and biases that result in the lowest possible loss are the optimal values that the model should use to make predictions.

In summary, gradient descent is an optimization algorithm that is used to find the values of the weights and biases that minimize the loss function in a machine learning model. It involves adjusting the values of the weights and biases in small increments based on the gradient of the loss function until the loss is minimized or the model has reached a satisfactory level of performance.

<figure class="center">
    <img src="/img/postimages/dnn.png" alt="Deep Neural Network" width="100%" height="350px" >
    <figcaption><pre></pre></figcaption>
</figure>

A Deep Neural Network consists of multiple hidden layers. A neural network with more than one hidden layer is consider deep.

Deep neural networks, which have many layers, have been shown to be more powerful and effective than shallow neural networks, which have fewer layers, for many tasks such as image and speech recognition.

Deep networks are able to learn hierarchical representations of the data, where lower layers learn simple patterns and higher layers learn increasingly complex and abstract patterns based on the outputs of the lower layers.

Additionally, deep networks can be trained more efficiently using techniques such as backpropagation and batch normalization, which helps to improve the performance of the network.

---

# _References_

[^1]: [Neural Networks and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning "Neural Networks and Deep Learning on Coursera")
