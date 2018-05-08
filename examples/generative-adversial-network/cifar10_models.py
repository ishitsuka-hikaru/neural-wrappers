import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from neural_wrappers.models import MobileNetV2Cifar10
from neural_wrappers.pytorch import NeuralNetworkPyTorch

class GeneratorConvTransposed(NeuralNetworkPyTorch):
	def __init__(self, inputSize, outputSize):
		super().__init__()
		self.inputSize = inputSize
		self.outputSize = outputSize

		self.conv1 = nn.ConvTranspose2d(in_channels=inputSize, out_channels=512, kernel_size=4,\
			stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(512)
		self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, \
			stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, \
			stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, \
			stride=2, padding=1, bias=False)

	def forward(self, x):
		x = x.view(-1, self.inputSize, 1, 1)
		y1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
		y2 = F.leaky_relu(self.bn2(self.conv2(y1)), 0.2)
		y3 = F.leaky_relu(self.bn3(self.conv3(y2)), 0.2)
		y4 = F.tanh(self.conv4(y3))
		y4 = y4.view(-1, *self.outputSize)
		return y4

class DiscriminatorMobileNetV2(MobileNetV2Cifar10):
	def __init__(self, outputSize):
		super().__init__(num_classes=1)

	def forward(self, x):
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layers(out)
		out = F.relu(self.bn2(self.conv2(out)))
		# NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		out = F.sigmoid(out)
		out = out.view(out.shape[0])
		return out