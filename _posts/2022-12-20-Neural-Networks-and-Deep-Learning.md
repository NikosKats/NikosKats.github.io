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

# This is a short summary for Neural Networks and Deep Learning course by Andrew Ng.

## Introduction to Deep Learning

A **Neuron** is inspired by the biological neuron and is the basic building block of neural networks. Are responsible for computing and transmitting information through the network.

A neuron consists of a set of inputs, called dendrites, and a single output, called an axon. The inputs of a neuron are connected to the outputs of other neurons, forming a network of connections. Each connection is associated with a weight, which determines the strength of the connection.

<figure class="center">
    <img src="/img/postimages/neuralNetwork.jpg" alt="Shallow Neural Network" width="350px" height="350px" >
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
    <img src="/img/postimages/logisticRegression.png" alt="Shallow Neural Network" width="350px" height="350px" >
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
                <figcaption>A shallow neural network consists of an input, a hidden and an output layer.</figcaption>
            </figure>
        </div>
        <div class="column" style="background-color:#ffffff;">
            <p hidden>hidden paragraph</p>
        </div>
        <div class="column" style="background-color:#ffffff;">
            <figure>
            <img src="/img/postimages/activationFunctions.png"
                alt="Activation Functions">
            <figcaption>Activation functions are mathematical equations that determine the output of a neuron in a neural network.</figcaption>
        </figure>
        </div>
    </div>

---

# _References_

[^1]: [Neural Networks and Deep Learning by Andrew Ng](https://www.coursera.org/learn/neural-networks-deep-learning "Neural Networks and Deep Learning on Coursera")
