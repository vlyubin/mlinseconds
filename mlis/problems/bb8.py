import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

NUM_FEATURES = 4

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
            x_featurized = torch.zeros([x.size(0), NUM_FEATURES], dtype=torch.float32)
            # Make an array having 2*item[row,i] + item[row,i+1] sums
            special = (2 * x[:, :-1] + x[:, 1:]).data.numpy()
            for i in range(NUM_FEATURES):
                counts = np.where(special == i, np.ones(special.shape),
                                  np.zeros(special.shape)).sum(axis=1)
                x_featurized[:, i] += torch.FloatTensor(counts)
            return self.net(x_featurized)

        return self.net(x)

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

        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 2

    # Return trained model
    def train_model(self, train_data, train_target, context):
        print('Featurizing train data!')
        train_data_featurized = torch.zeros([train_data.size(0), NUM_FEATURES], dtype=torch.float32)

        # Make an array having 2*item[row,i] + item[row,i+1] sums
        special = (2 * train_data[:, :-1] + train_data[:, 1:]).data.numpy()

        # Effectively train_data_featurized is just 4 numbers for each test - number of 00, 01, 10, 11
        # in the sequence. Since state transitions only depend on the previous number, this gives
        # a good guidance of what's happening.
        for i in range(NUM_FEATURES):
            counts = np.where(special == i, np.ones(special.shape), np.zeros(special.shape)).sum(axis=1)
            train_data_featurized[:, i] += torch.FloatTensor(counts)

        BATCH_SIZE = 256
        print('Training started!')
        model = SolutionModel(train_data_featurized.size(1), 1, self)

        if torch.cuda.is_available():
            model.cuda()
            train_data_featurized = train_data_featurized.cuda()
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
            batch = train_data_featurized[batch_idx:batch_idx + BATCH_SIZE, :]
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
            # calculate error
            loss = model.calc_error(output, train_target[batch_idx:batch_idx + BATCH_SIZE])

            if time_left < 0.15:
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
        if step % 25 == 0:
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
        self.test_limit = 0.8

# There are 2 languages.
class Language:
    def __init__(self, states_count, letters_count):
        self.states_count = states_count
        with torch.no_grad():
            self.state_to_state_prob = torch.FloatTensor(states_count, states_count).uniform_()
            self.state_to_state_prob = self.state_to_state_prob/torch.sum(self.state_to_state_prob, dim=1).view(-1,1)
            self.state_to_state_cum_prob = torch.cumsum(self.state_to_state_prob, dim=1)

    def balance(self, sentences):
        letters_id, counts = torch.unique(sentences.view(-1), sorted=True, return_counts=True)
        perm = torch.randperm(letters_id.size(0))
        letters_id = letters_id[perm]
        counts = counts[perm]
        total = counts.sum().item()
        x = torch.ByteTensor(total+1).zero_()
        x[0] = 1
        xs = [x]
        for letter_id, count in zip(letters_id, counts):
            cc = count.item()
            nx = xs[-1].clone()
            nx[cc:][xs[-1][:-cc]] = 1
            xs.append(nx)
        best_balance = total//2
        while xs[-1][best_balance].item() == 0:
            best_balance -= 1
        #if best_balance != total//2:
        #    print("UNBALANCED")
        current_balance = best_balance
        balance_set = [False for _ in range(letters_id.size(0))]
        last_ind = len(xs)-1
        while current_balance != 0:
            while xs[last_ind-1][current_balance].item() == 1:
                last_ind -= 1
            balance_set[last_ind-1] = True
            current_balance -= counts[last_ind-1].item()
        b_sentences = sentences.clone()
        self.state_to_state_letter = self.state_to_state_letter.view(-1)
        for ind, set_id in enumerate(balance_set):
            val = 0
            if set_id:
                val = 1
            b_sentences[sentences == letters_id[ind]] = val
            self.state_to_state_letter[letters_id[ind]] = val
        assert b_sentences.view(-1).sum() == best_balance
        self.state_to_state_letter = self.state_to_state_letter.view(self.states_count, self.states_count)
        return b_sentences

    def gen(self, count, length):
        with torch.no_grad():
            self.state_to_state_letter = torch.arange(self.states_count*self.states_count).view(self.states_count, self.states_count)
            #self.state_to_state_letter.random_(0,2)
            sentences = torch.LongTensor(count, length)
            states = torch.LongTensor(count).random_(0, self.states_count)
            for i in range(length):
                res = torch.FloatTensor(count).uniform_()
                probs = self.state_to_state_cum_prob[states]
                next_states = self.states_count-(res.view(-1,1) < probs).sum(dim=1)
                next_states = next_states.clamp(max=self.states_count-1)
                letters_ind = self.state_to_state_letter[states, next_states]
                sentences[:,i] = letters_ind
                states = next_states
            sentences = self.balance(sentences)
            return sentences

    def calc_probs(self, sentences):
        size = sentences.size(0)
        states_count = self.state_to_state_prob.size(0)
        length = sentences.size(1)
        with torch.no_grad():
            state_to_prob = torch.FloatTensor(size, states_count).double()
            state_to_prob[:,:] = 1.0
            for i in range(length):
                letters = sentences[:,i]
                s1 = self.state_to_state_letter.size()
                s2 = letters.size()
                sf = s2+s1

                t1 = self.state_to_state_letter.view((1,)+s1).expand(sf)
                t2 = letters.view(s2+(1,1)).expand(sf)
                t3 = self.state_to_state_prob.view((1,)+s1).expand(sf).double()
                t4 = (t1 == t2).double()
                t5 = torch.mul(t3, t4)
                t6 = state_to_prob
                next_state_to_prob = torch.matmul(t6.view(t6.size(0), 1, t6.size(1)), t5).view_as(t6)
                state_to_prob = next_state_to_prob
            return state_to_prob.sum(dim=1)

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, length, states_count, letters_count, seed):
        while True:
            torch.manual_seed(seed)
            languages = [Language(states_count, letters_count), Language(states_count, letters_count)]
            data_size_per_lang = data_size//len(languages)
            datas = []
            targets = []
            for ind, lan in enumerate(languages):
                datas.append(lan.gen(data_size_per_lang, length))
                t = torch.LongTensor(data_size_per_lang)
                t[:] = ind
                targets.append(t)
            bad_count = 0
            good_count = 0
            for ind, data in enumerate(datas):
                probs = [lan.calc_probs(data) for lan in languages]
                bad_count += (probs[ind] <= probs[1-ind]).long().sum().item()
                good_count += (probs[ind] > probs[1-ind]).long().sum().item()
            best_prob = good_count/(bad_count+good_count)
            if best_prob > 0.95:
                break
            print("Low best prob = {}, seed = {}".format(best_prob, seed))
            seed += 1

        data = torch.cat(datas, dim=0)
        target = torch.cat(targets, dim=0)
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data, target.view(-1, 1).float(), best_prob)

    def create_case_data(self, case):
        data_size = 256*4
        case_configs = [
                (8, 2, 7),
                (16, 3, 34),
                (32, 4, 132),
                (64, 5, 13),
                (128, 6, 1),
                (256, 7, 5),
                (256, 7, 6),
                (256, 7, 71),
                (256, 7, 19),
                (256, 7, 40)
                ]
        case_config = case_configs[min(case, 10)-1]
        length = case_config[0]
        states_count = case_config[1]
        # seed help generate data faster
        seed = 1000*case + case_config[2]
        letters_count = 2
        data, target, best_prob = self.create_data(2*data_size, length, states_count, letters_count, seed)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("States = {} Length = {} Seed = {} Best prob = {:.3}".format(states_count, length, seed, best_prob))

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
    gs.GridSearch().run(Config(), case_number=0, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
