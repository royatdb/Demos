# Databricks notebook source
# MAGIC %md ## PyTorch Demo
# MAGIC 
# MAGIC This notebook demonstrates how to use PyTorch on the Spark driver node to fit a neural network on MNIST handwritten digit recognition data.
# MAGIC 
# MAGIC 
# MAGIC The content of this notebook is [copied from the PyTorch project](https://github.com/pytorch/examples/blob/53f25e0d0e2710878449900e1e61d31d34b63a9d/mnist/main.py) under the [license](https://github.com/pytorch/pytorch/blob/a90c259edad1ea4fa1b8773e3cb37240df680d62/LICENSE) with slight modifications in comments. Thanks to the developers of PyTorch for this example.

# COMMAND ----------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# COMMAND ----------

# MAGIC %md ### Handwritten Digit Recognition
# MAGIC 
# MAGIC This tutorial walks through a classic computer vision application of identifying handwritten digits. We will train a simple Convolutional Neural Network on the MNIST dataset.

# COMMAND ----------

# MAGIC %md ### Parameters
# MAGIC We encourage using a GPU cluster to run this notebook. By setting `cuda=True`, `PyTorch` will take advantage of CUDA enabled GPU to accelarate computation. 
# MAGIC 
# MAGIC - `batch_size`: number of examples in a training mini-batch
# MAGIC - `test_batch_size`: number of examples in a testing/inference mini-batch. This is usually larger than `batch_size` since we don't have to do backward pass during testing/inference.
# MAGIC 
# MAGIC The training algorithm will go through the training set for `epochs` passes. 
# MAGIC We use Stochastic Gradient Descent with learning rate `lr` and momentum factor `momentum`.
# MAGIC Please check [PyTorch SGD implementation](https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py) for details.

# COMMAND ----------

MNIST_DIR = "/tmp/data/mnist"

Params = namedtuple('Params', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'momentum', 'seed', 'cuda', 'log_interval'])
args = Params(batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, momentum=0.5, seed=1, cuda=False, log_interval=200)

# COMMAND ----------

# MAGIC %md ### Prepare MNIST Dataset 
# MAGIC We download the dataset, shuffle the rows, create batches and standardize the features.

# COMMAND ----------

torch.manual_seed(args.seed)

data_transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_DIR, train=True, download=True,
                       transform=data_transform_fn),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_DIR, train=False, 
                       transform=data_transform_fn),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)


# COMMAND ----------

# MAGIC %md ### Construct a CNN model
# MAGIC Now we create a simple CNN model with two *convolutional* layers (conv) and two *fully connected* (fc) layers. We also add a *dropout* layer between the conv and fc layers.

# COMMAND ----------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
      
model = Net()
model.share_memory() # gradients are allocated lazily, so they are not shared here

# COMMAND ----------

# MAGIC %md ### Model Training
# MAGIC To train the model, let us define a *Negative Log Likelihood* loss and create a Stochastic Gradient Descent optimizer with *momentum*.
# MAGIC Calling `loss.backward()` followed by `optimizer.step()` updates the model parameters.

# COMMAND ----------

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()      
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()      
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


# Run the training loop over the epochs (evaluate after each)
if args.cuda:
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
for epoch in range(1, args.epochs + 1):
    train_epoch(epoch, args, model, train_loader, optimizer)
    test_epoch(model, test_loader)    