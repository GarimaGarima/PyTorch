import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from visdom import Visdom


# Creating a neural network with 2 hidden layer convoluted and pooling layer
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)  # first layer
        self.linear2 = nn.Linear(100, 50)  # pooling layer
        self.final_linear = nn.Linear(50, 10)  # output layer

        self.relu = nn.ReLU()  # activation function

    def forward(self, images):
        x = images.view(-1, 28 * 28)  # flatten the image
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final_linear(x)
        return x


# transforming the data or images in one format using transform
# after transforming , normalizing the data with 0.5 mean and 0.5 standard deviation
t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                                    ])

# downloading the mnist data set
mnist_data = torchvision.datasets.MNIST('mnist_data', transform=t, download=True)

# creating an iterator for mnist downloaded data and defining number of images to use in one iteration
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=128)

# training the model
model = Mnet()
cec_loss = nn.CrossEntropyLoss()  # loss function
params = model.parameters()  # hyperparameters and gradiant
optimizer = optim.Adam(params=params, lr=0.001)

n_epochs = 3
n_iterations = 0
vis = Visdom()
vis_window = vis.line(np.array([0]), np.array([0]))

for e in range(n_epochs):
    for i, (images, labels) in enumerate(mnist_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)

        model.zero_grad()
        loss = cec_loss(output, labels)
        loss.backward()

        optimizer.step()

        n_iterations += 1

        vis.line(np.array([loss.item()]), np.array([n_iterations]), win=vis_window, update='append')
