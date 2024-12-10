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
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        hidden_size = 256
        output_size = 10

        # Define layers
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Forward pass: Input -> Hidden Layer with ReLU -> Output Layer
        hidden = relu(self.fc1(x))
        logits = self.fc2(hidden)  # No activation on output layer
        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        # Forward pass to get logits
        logits = self.run(x)
        
        # Convert one-hot labels to class indices
        targets = y.argmax(dim=1)
        
        # Compute cross-entropy loss
        loss = cross_entropy(logits, targets)
        return loss

    def train(self, dataset):
        """
        Trains the model.

        Trains the model until it reaches a high validation accuracy or
        until it hits the maximum number of epochs.

        Inputs:
            dataset: The training dataset which should provide a
                     get_validation_accuracy(model) method.
        """
        # Create a DataLoader for batching
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Set a validation accuracy goal and maximum number of epochs
        target_val_accuracy = 0.975
        max_epochs = 30

        # Set the model to training mode
        self.train()

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Compute the loss
                loss = self.get_loss(x, y)
                
                # Backpropagate the loss
                loss.backward()
                
                # Update the parameters
                self.optimizer.step()
                
                # Accumulate the loss
                epoch_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
            
            # Check validation accuracy
            val_accuracy = dataset.get_validation_accuracy(self)
            print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
            
            # Stop training if target validation accuracy is achieved
            if val_accuracy > target_val_accuracy:
                print("Target validation accuracy reached. Stopping training.")
                break