# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):

    def __init__(self):
        super(SolutionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        if not self.is_training and torch.cuda.is_available():
            x = x.cuda()
      
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        rv = F.log_softmax(x, dim=1)

        if not self.is_training and torch.cuda.is_available():
            rv = rv.cpu()

        return rv


    def calc_error(self, output, target):
        return F.nll_loss(output, target)

    def calc_predict(self, x):
        return torch.argmax(x, dim=1)

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.01
        # Grid search settings, see grid_search_tutorial
        self.learning_rate_grid = []
        self.hidden_size_grid = []
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 2

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
#        return self.grid_search_tutorial()
        # Model represent our neural network

        BATCH_SIZE = 256
        print('Training started!')
        model = SolutionModel()

        if torch.cuda.is_available():
            model.cuda()
            train_data = train_data.cuda()
            train_target = train_target.cuda()

        # Optimizer used for training neural network
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        batch_idx = 0
        model.is_training = True
        
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # train on a batch
            batch = train_data[batch_idx:batch_idx + BATCH_SIZE,:]
            # evaluate model => model.forward(data)
            output = model(batch)
            # if x < 0.5 predict 0 else predict 1
            # Number of correct predictions
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(train_target[batch_idx:batch_idx + BATCH_SIZE].view_as(pred)).long().sum().item()
            # Total number of needed predictions
            total = pred.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            # calculate error 
            loss = model.calc_error(output, train_target[batch_idx:batch_idx + BATCH_SIZE])

            if time_left < 0.1:
                self.print_stats(context.step, loss, correct, total, force=True)
                break

            loss.backward()
            # print progress of the learning
            self.print_stats(context.step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            # increment batch index
            batch_idx += BATCH_SIZE
            if batch_idx >= train_data.size(0):
                batch_idx = 0

        model.is_training = False
        print('Training is done!')
        return model

    def print_stats(self, step, error, correct, total, force=False):
        if step % 25 == 0 or force:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 25000
        self.test_limit = 0.95

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0,mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0,1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=4)
