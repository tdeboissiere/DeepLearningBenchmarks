import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg
import utils


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        # First conv block
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        # Second conv block
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)

        # Flatten
        out = out.view(out.size(0), -1)

        # Linear
        out = F.relu(self.fc1(out))
        out = F.log_softmax(self.fc2(out))

        return out


def run_SimpleCNN(batch_size, nb_epoch):

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    list_transforms = [transforms.ToTensor(), normTransform]
    trainTransform = transforms.Compose(list_transforms)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset = dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
    trainLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    net = SimpleCNN()

    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(nb_epoch):

        s = time.time()

        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        print time.time() - s


def run_VGG16(batch_size, n_trials):

    # Initialize network
    net = vgg.vgg16()
    net.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Data
    n_classes = 1000
    labels = np.random.randint(0, 1000, batch_size * n_trials).astype(np.uint8).tolist()
    labels = torch.LongTensor(labels)
    inputs = torch.randn(batch_size * n_trials, 3, 224, 224)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    t0 = time.time()
    n = 0
    for i, (X, y) in enumerate(dataloader):

        ll = Variable(y.cuda(async=True))
        inp = Variable(X.cuda(async=True))

        # forward pass
        outputs = net(inp)

        # compute loss
        loss = criterion(outputs, ll)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        n += 1

    t1 = time.time()

    # Print summary
    utils.print_module("pytorch version: %s" % torch.__version__)
    utils.print_result("%7.3f ms." % (1000. * (t1 - t0) / n_trials))