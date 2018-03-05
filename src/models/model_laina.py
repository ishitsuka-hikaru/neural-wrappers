from .resnet50_notop import ResNet50NoTop
from .upsample import UpSampleLayer
from wrappers.pytorch import NeuralNetworkPyTorch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

# Implementation of the Laina model from https://arxiv.org/abs/1606.00373
class ModelLaina(NeuralNetworkPyTorch):
	def __init__(self, labelShape, baseModelType, upSampleType):
		super().__init__()
		assert baseModelType in ("resnet50", )
		assert upSampleType in ("unpool", "bilinear", "nearest", "conv_transposed")
		self.labelShape = labelShape
		self.baseModelType = baseModelType
		self.upSampleType = upSampleType

		if self.baseModelType == "resnet50":
			self.baseModel = ResNet50NoTop()

		self.conv_3_1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
		self.bn_3_1 = nn.BatchNorm2d(1024)
		self.upConv_3_2 = UpSampleLayer(inShape=(8, 10), dIn=1024, dOut=512, Type=upSampleType)
		self.upConv_3_3 = UpSampleLayer(inShape=(16, 20), dIn=512, dOut=256, Type=upSampleType)
		self.upConv_3_4 = UpSampleLayer(inShape=(32, 40), dIn=256, dOut=128, Type=upSampleType)
		self.upConv_3_5 = UpSampleLayer(inShape=(64, 80), dIn=128, dOut=64, Type=upSampleType)
		self.conv_3_6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3)

	def __str__(self):
		return "Laina. Base model: %s. Upsample type: %s. Label shape: %s" % (self.baseModelType, \
			self.upSampleType, self.labelShape)

	def forward(self, x):
		# Move depth first (MB, 228, 304, 3) => (MB, 3, 228, 304)
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)

		# Output of ResNet-50 is (MB, 2048, 8, 10), now we just add the 3rd row in the paper
		y_base = self.baseModel(x)

		# 3rd row of paper
		y_3_1 = F.relu(self.bn_3_1(self.conv_3_1(y_base)))
		y_3_2 = self.upConv_3_2(y_3_1)
		y_3_3 = self.upConv_3_3(y_3_2)
		y_3_4 = self.upConv_3_4(y_3_3)
		y_3_5 = self.upConv_3_5(y_3_4)
		y_3_6 = F.pad(self.conv_3_6(y_3_5), (1, 1, 1, 1), "reflect")
		# y_3_6 shape = (N, 1, 128, 160) => (N, 128, 160)
		y_3_6 = y_3_6.view(y_3_6.shape[0], y_3_6.shape[2], y_3_6.shape[3])

		assert tuple(y_3_6.shape[1 : 3]) == self.labelShape
		return y_3_6