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

		self.dropout = nn.Dropout2d(0.2)
		self.conv1 = nn.ConvTranspose2d(in_channels=inputSize, out_channels=512, kernel_size=3, \
			stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(512)
		self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, \
			stride=2, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, \
			stride=2, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, \
			stride=2, padding=0, bias=False)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, \
			stride=1, padding=0, bias=False)
		self.bn5 = nn.BatchNorm2d(32)
		self.conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=1, \
			stride=1, padding=0, bias=False)

	def forward(self, x):
		x = x.view(-1, self.inputSize, 1, 1)
		y1 = F.leaky_relu(self.bn1(self.dropout(self.conv1(x))), 0.2)
		y2 = F.leaky_relu(self.bn2(self.dropout(self.conv2(y1))), 0.2)
		y3 = F.leaky_relu(self.bn3(self.dropout(self.conv3(y2))), 0.2)
		y4 = F.leaky_relu(self.bn4(self.dropout(self.conv4(y3))), 0.2)
		y5 = F.leaky_relu(self.bn5(self.dropout(self.conv5(y4))), 0.2)
		y6 = F.tanh(self.conv6(y5))
		y6 = tr.transpose(tr.transpose(y6, 1, 3), 1, 2).contiguous()
		return y6

class DiscriminatorConv(NeuralNetworkPyTorch):
	def __init__(self, outputSize):
		super().__init__()
		self.outputSize = outputSize
		self.dropout = nn.Dropout2d(0.2)
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.fc5 = nn.Linear(256 * 4 * 4, 1)

	def forward(self, x):
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)

		y1 = F.leaky_relu(self.bn1(self.dropout(self.conv1(x))))
		y2 = F.leaky_relu(self.bn2(self.dropout(self.conv2(y1))))
		y3 = F.leaky_relu(self.bn3(self.dropout(self.conv3(y2))))
		y4 = F.leaky_relu(self.bn4(self.dropout(self.conv4(y3))))
		y4 = y4.view(-1, 256 * 4 * 4)
		y5 = F.sigmoid(self.fc5(y4))[..., 0]
		# y5 = y5.view(y5.shape[0])
		return y5