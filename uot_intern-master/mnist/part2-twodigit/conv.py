import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        # TODO initialize model layers here
        self.conv_1 = nn.Conv2d(1, out_channels=32, kernel_size=3)
        self.conv_2 = nn.Conv2d(32, out_channels=64, kernel_size=3)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2880,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.out = nn.Linear(64,20)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)

    def forward(self, x):

        #conv1
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.max_pool2d(x)

        #conv2
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = self.max_pool2d(x)

        #faltten
        xf = self.flatten(x)

        #fc1
        xf = self.fc1(xf)
        xf = F.relu(xf)
        xf = self.dropout(xf)

        #fc2
        xf = self.fc2(xf)
        xf = F.relu(xf)
        xf = self.dropout(xf)

        #out
        xf = self.out(xf)
        split0, split1 = xf.split(10, dim=1)
        #out_first_digit = F.softmax(split0)
        #out_first_digit = F.softmax(split1)

        out_first_digit = split0
        out_second_digit = split1


        # TODO use model layers to predict the two digits

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
