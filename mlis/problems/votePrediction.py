# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
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

        self.net = nn.Sequential(
            self.linear1,
            torch.nn.LeakyReLU(0.01),
            self.batch_norm1,

            self.linear2,
            torch.nn.LeakyReLU(0.01),
            self.batch_norm2,

            self.linear3,
            torch.nn.LeakyReLU(0.01),
            self.batch_norm3,

            self.linear4,
            self.batch_norm4,
            torch.nn.Sigmoid()
        )

        self.loss = nn.BCELoss()

    def forward(self, x):
        if not self.is_training:
            x = x.view(-1, 8)
        return self.net(x)

    def calc_error(self, output, target):
        # Effectively compute sum for the entire case and use that for loss
        output = output.view(-1, self.num_voters)
        output = torch.sum(output, dim=1)
        output /= self.num_voters
        # This is loss function
        error = self.loss(output.view(-1), target)
        return error.sum()

    def calc_predict(self, x):
        x = x.round()
        x = x.view(-1, self.num_voters)
        x = torch.sum(x, dim=1)
        x /= (self.num_voters)
        # I found that having a number slightly greater than 0.5 (so not using .round()) helps here
        x[x > 0.52] = 1
        x[x <= 0.52] = 0
        return x

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.007
        # Control number of hidden neurons. "Why 42?" you might ask. Well, read this:
        # https://en.wikipedia.org/wiki/42_(number)#The_Hitchhiker's_Guide_to_the_Galaxy
        self.hidden_size = [32, 42, 42]

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
        num_voters = train_data.size(1) // 8
        print("Num voters %d" % num_voters)

        train_target = train_target.view(-1)

        print('Num 0s ', (train_target == 0).sum().item())
        print('Num 1s ', (train_target == 1).sum().item())

        # Split train_data into train_data_0 and train_data_1
        train_data_0 = train_data[train_target == 0, :]
        train_data_1 = train_data[train_target == 1, :]
        train_target_0 = train_target[train_target == 0]
        train_target_1 = train_target[train_target == 1]

        train_data_0 = train_data_0.view(-1, 8)
        train_data_1 = train_data_1.view(-1, 8)

        HALF_BATCH_SIZE = 384
        print('Training started!')

        model = SolutionModel(8, 1, self)
        model.is_training = True
        model.num_voters = num_voters

        # Optimizer used for training neural network
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)

        batch_idx_0 = 0
        batch_idx_1 = 0

        correct_streak = 0

        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # Stich a batch from 0 and 1 examples so that we have equal number of each
            batch = torch.cat((train_data_0[batch_idx_0*num_voters:batch_idx_0*num_voters + HALF_BATCH_SIZE*num_voters,:],
                               train_data_1[batch_idx_1*num_voters:batch_idx_1*num_voters + HALF_BATCH_SIZE*num_voters,:]),
                              0)
            target = torch.cat((train_target_0[batch_idx_0:batch_idx_0 + HALF_BATCH_SIZE],
                                train_target_1[batch_idx_1:batch_idx_1 + HALF_BATCH_SIZE]),
                              0)
            # evaluate model => model.forward(data)
            output = model(batch)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)

            if correct == total:
                correct_streak += 1
            else:
                correct_streak = 0

            # No more time left or got perfect score 5 times in a row - stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct_streak == 5:
                self.print_stats(context.step, error, correct, total, force=True)
                break
            # calculate error
            error = model.calc_error(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            # increment batch indices
            batch_idx_0 += HALF_BATCH_SIZE
            if batch_idx_0 * num_voters >= train_data_0.size(0):
                batch_idx_0 = 0
            batch_idx_1 += HALF_BATCH_SIZE
            if batch_idx_1 * num_voters >= train_data_1.size(0):
                batch_idx_1 = 0

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

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

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
