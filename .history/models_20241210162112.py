from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        
        self.w = Parameter(ones(1, dimensions)) #Initialize your weights here

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

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return tensordot(self.w, x)
        


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if self.run(x) >= 0 else -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            errorCount = 1
            while(errorCount > 0):
                errorCount = 0
                for batch in dataloader:
                    if self.get_prediction(batch['x']) != batch['label']:
                        self.w += batch['x'] * batch['label']
                        errorCount += 1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        self.layer1 = Linear(1, 200)
        self.layer2 = Linear(200, 400)
        self.layer3 = Linear(400, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        h1 = relu(self.layer1(x))
        h2 = relu(self.layer2(h1))
        h3 = self.layer3(h2)
        return h3
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predictions = self.forward(x)
        return mse_loss(predictions, y)
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        data = DataLoader(dataset, batch_size=64)
       
        for epoch in range(1000):
            epoch_loss = 0.0
            for batch in data:
                x, y = batch['x'], batch['label'] 

                self.optimizer.zero_grad()

                loss = self.get_loss(x, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if epoch_loss < 0.02:
                break

class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector. Each entry is a float between 0 and 1.

    The model should output a (batch_size x 10) tensor of scores (logits).
    We will use cross_entropy as the loss, which requires integer class labels.
    Since the dataset labels are given as one-hot vectors, we will convert them
    to integer class indices before using cross_entropy.

    Goal: achieve at least 97% on the test set.
    """
    def __init__(self):
        super().__init__()
        input_size = 28 * 28  
        h1 = 256
        h2 = 128
        h3 = 64
        output_size = 10

        self.fc1 = Linear(input_size, h1)
        self.fc2 = Linear(h1, h2)
        self.fc3 = Linear(h2, h3)
        self.fc4 = Linear(h3, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Input: x of shape (batch_size x 784)
        Output: logits of shape (batch_size x 10)
        """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: (batch_size x 784)
            y: (batch_size x 10), one-hot vectors for correct classes

        We must convert y from one-hot encoding to integer class indices
        for cross_entropy.
        """
        logits = self.run(x)
        targets = y.argmax(dim=1)
        loss = cross_entropy(logits, targets)
        return loss

    def train(self, dataset):
        """
        Trains the model.

        We'll:
        1. Create a DataLoader to iterate over the training data in batches.
        2. Run several epochs, each epoch going through all training batches.
        3. After each epoch, check validation accuracy.
        4. Stop early if validation accuracy surpasses a chosen threshold.
        """
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        target_val_acc = 0.98
        max_epochs = 30  

        for epoch in range(max_epochs):
            for batch in dataloader:
                x, y = batch['x'], batch['label']

                self.optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                self.optimizer.step()

            val_acc = dataset.get_validation_accuracy()
            # Print progress if desired (you can remove this in final code)
            # print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")

            if val_acc >= target_val_acc:
                break