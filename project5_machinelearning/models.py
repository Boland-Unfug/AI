import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # calculate the dot product of the weights and the input
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # if the dot product is greater than 0, return 1, else return -1
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # initialize the accuracy to 0
        accuracy = 0
        count = 0
        # while the accuracy is not 100%
        for x, y in dataset.iterate_once(1):
            count += 1
        while accuracy != 1:
            # initialize the accuracy to 0
            accuracy = 0
            # for each x, y in the dataset
            for x, y in dataset.iterate_once(1):
                # if the prediction is equal to the actual value
                if self.get_prediction(x) == nn.as_scalar(y):
                    # increase the accuracy
                    accuracy += 1
                # else if the prediction is not equal to the actual value
                else:
                    # update the weights
                    self.w.update(x, nn.as_scalar(y))

            # divide the accuracy by the number of data points in the dataset, cant use get_num_examples() because it is not defined
            accuracy = accuracy / count


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # here is a list of tried values for the weights and biases
        # self.w1 = nn.Parameter(1, 100)
        # self.b1 = nn.Parameter(1, 100)
        # self.w2 = nn.Parameter(100, 1)
        # self.b2 = nn.Parameter(1, 1)
        # Learnign rate     results:
        # -0.01             0.019998
        # -0.05             0.019906 (best)
        # -0.1              0.019994
        # -0.5              failure
        # self.w1 = nn.Parameter(1, 50)
        # self.b1 = nn.Parameter(1, 50)
        # self.w2 = nn.Parameter(50, 1)
        # self.b2 = nn.Parameter(1, 1)
        # Learnign rate     results:
        # -0.01             0.019998
        # -0.05             0.019961
        # -0.1              0.019989
        # -0.5              failure
        # self.w1 = nn.Parameter(1, 200)
        # self.b1 = nn.Parameter(1, 200)
        # self.w2 = nn.Parameter(200, 1)
        # self.b2 = nn.Parameter(1, 1)
        # Learnign rate     results:
        # -0.01             0.019998
        # -0.05             0.019982
        # -0.1              0.019994
        # -0.5              failure
        # initialize the learning rate
        self.learning_rate = -0.05
        # Using the best weights and biases from the above tests
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # ensure that x is a node
        if type(x) != nn.Constant:
            x = nn.Constant(x)
        # calculate the dot product of the weights and the input, this needs to input node objects
        # print("X", type(x))
        # print("w1", type(self.w1))
        # print(x)
        dot_product = nn.Linear(x, self.w1)
        # add the bias
        add_bias = nn.AddBias(dot_product, self.b1)
        # calculate the relu
        relu = nn.ReLU(add_bias)
        # calculate the dot product of the weights and the relu
        dot_product2 = nn.Linear(relu, self.w2)
        # add the bias
        add_bias2 = nn.AddBias(dot_product2, self.b2)
        # return the result
        return add_bias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # calculate the loss
        if type(x) != nn.Constant:
            x = nn.Constant(x)
        if type(y) != nn.Constant:
            y = nn.Constant(y)

        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # while the loss is greater than 0.02
        while nn.as_scalar(self.get_loss(dataset.x, dataset.y)) >= 0.02:
            # calculate the gradient
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(
                self.get_loss(dataset.x, dataset.y),
                [self.w1, self.b1, self.w2, self.b2],
            )
            # update the weights and biases
            self.w1.update(grad_w1, self.learning_rate)
            self.b1.update(grad_b1, self.learning_rate)
            self.w2.update(grad_w2, self.learning_rate)
            self.b2.update(grad_b2, self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initialize the learning rate
        # also tested with -.75, -0.5, -0.25, -0.1
        self.learning_rate = -0.75
        # Using these weights:
        # Hidden layer size 200
        # Batch size 100
        # Learning rate 0.5
        # One hidden layer (2 linear layers in total)

        self.w1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # ensure that x is a node
        if type(x) != nn.Constant:
            x = nn.Constant(x)
        # calculate the dot product of the weights and the input, this needs to input node objects
        dot_product = nn.Linear(x, self.w1)
        # add the bias
        add_bias = nn.AddBias(dot_product, self.b1)
        # calculate the relu
        relu = nn.ReLU(add_bias)
        # calculate the dot product of the weights and the relu
        dot_product2 = nn.Linear(relu, self.w2)
        # add the bias
        add_bias2 = nn.AddBias(dot_product2, self.b2)
        # return the result
        return add_bias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # calculate the loss
        if type(x) != nn.Constant:
            x = nn.Constant(x)
        if type(y) != nn.Constant:
            y = nn.Constant(y)

        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # while the accuracy is less than 0.975
        while dataset.get_validation_accuracy() < 0.975:
            print(dataset.get_validation_accuracy())
            # calculate the gradient
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(
                self.get_loss(dataset.x, dataset.y),
                [self.w1, self.b1, self.w2, self.b2],
            )
            # update the weights and biases
            self.w1.update(grad_w1, self.learning_rate)
            self.b1.update(grad_b1, self.learning_rate)
            self.w2.update(grad_w2, self.learning_rate)
            self.b2.update(grad_b2, self.learning_rate)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initialize the learning rate

        self.learning_rate = -0.1
        # Using these weights:
        # Hidden layer size 200
        # Batch size 100
        # Learning rate 0.5
        # One hidden layer (2 linear layers in total)

        self.w1 = nn.Parameter(self.num_chars, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, len(self.languages))
        self.b2 = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # print("xs", xs)
        # start by initializing the first hidden layer
        # calculate the dot product of the weights and the input, this needs to input node objects
        dot_product = nn.Linear(xs[0], self.w1)
        # add the bias
        add_bias = nn.AddBias(dot_product, self.b1)
        # calculate the relu
        relu = nn.ReLU(add_bias)
        # calculate the dot product of the weights and the relu
        dot_product2 = nn.Linear(relu, self.w2)
        # add the bias
        add_bias2 = nn.AddBias(dot_product2, self.b2)

        h1 = add_bias2

        # apply sub network that accepts a single character and produces a hidden state
        # but now depends on the previous hidden state
        # for each character in the word
        for i in range(1, len(xs)):
            # In other words, you should replace a computation of the form z0 = nn.Linear(x, W) with a computation of the form z = nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden)).
            # calculate the dot product of the weights and the input, this needs to input node objects
            dot_product = nn.Linear(xs[i], self.w1)
            # add the bias
            add_bias = nn.AddBias(dot_product, self.b1)
            # calculate the relu
            relu = nn.ReLU(add_bias)
            # calculate the dot product of the weights and the relu
            dot_product2 = nn.Linear(relu, self.w2)
            # add the bias
            add_bias2 = nn.AddBias(dot_product2, self.b2)

            h1 = nn.Add(h1, add_bias2)
        # print("SHAPE", h1)
        return h1

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # calculate the loss
        for x in xs:
            if type(x) != nn.Constant:
                x = nn.Constant(x)
        if type(y) != nn.Constant:
            y = nn.Constant(y)
        # print("returns ", nn.SoftmaxLoss(self.run(xs), y))
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # print(vars(dataset))
        # while the accuracy is less than 0.9
        # for data in dataset:
        while dataset.get_validation_accuracy() < 0.9:
            # calculate the gradient
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(
                self.get_loss(dataset.xs, dataset.y),
                [self.w1, self.b1, self.w2, self.b2],
            )
            # update the weights and biases
            self.w1.update(grad_w1, self.learning_rate)
            self.b1.update(grad_b1, self.learning_rate)
            self.w2.update(grad_w2, self.learning_rate)
            self.b2.update(grad_b2, self.learning_rate)
