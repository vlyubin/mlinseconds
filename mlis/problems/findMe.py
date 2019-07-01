# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
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
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size[0])
        self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.linear3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        self.linear4 = nn.Linear(self.hidden_size[2], output_size)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size[0], track_running_stats=False)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[1], track_running_stats=False)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[2], track_running_stats=False)
        self.batch_norm4 = nn.BatchNorm1d(output_size, track_running_stats=False)

        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.LeakyReLU(0.01)(x)
        x = self.batch_norm1(x)
        x = self.linear2(x)
        x = torch.nn.LeakyReLU(0.01)(x)
        x = self.batch_norm2(x)
        x = self.linear3(x)
        x = torch.nn.LeakyReLU(0.01)(x)
        x = self.batch_norm3(x)
        x = self.linear4(x)
        x = self.batch_norm4(x)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        error = self.loss(output, target)
        return error.sum()

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.01
        # Control number of hidden neurons
        self.hidden_size = [110, 50, 25]

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
        BATCH_SIZE = 768
        print('Training started!')
        print(train_data.size(), train_target.size())
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        model.is_training = True

        # Optimizer used for training neural network
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        batch_idx = 0
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
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target[batch_idx:batch_idx + BATCH_SIZE].view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.2:
                self.print_stats(context.step, error, correct, total, force=True)
                break
            # calculate error
            error = model.calc_error(output, train_target[batch_idx:batch_idx + BATCH_SIZE])
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
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
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

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
    sm.SolutionManager().run(Config(), case_number=-1)
