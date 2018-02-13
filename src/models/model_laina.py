from pytorch_wrapper import NeuralNetworkPyTorch, maybeCuda
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss import l2_loss, hubber_loss
import numpy as np

from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, model_urls
import torch.utils.model_zoo as model_zoo

# Class that loads a ResNet-50 module from torchvision and deletes the FC layer at the end, to use just as extractor
class ResNet50NoTop(ResNet):
	def __init__(self):
		super().__init__(Bottleneck, [3, 4, 6, 3])
		self.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
		self.fc = None
		self.avgpool = None

	# Overwrite the ResNet's forward phase to drop the average pooling and the fully connected layers
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

# Class that implements 3 methods for up-sampling from the bottom encoding layer
# "unpool" is the method described in Laina paper, with unpooling method with zeros + conv
# "nearest" and "bilinear" are based on using PyTorch's nn.Upsample layer
class UpSampleLayer(nn.Module):
	def __init__(self, inShape, dIn, dOut, Type):
		super(UpSampleLayer, self).__init__()
		assert Type in ("unpool", "bilinear", "nearest", "conv_transposed")
		assert len(inShape) == 2
		self.conv = nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=5)

		# This if is just so we can have common code of bilinear/nearest with the "unpooling" method. Some lambdas are
		#  used so the forward phase has no ifs, just some additional empty calls.
		if Type == "unpool":
			self.upSample = nn.MaxUnpool2d(kernel_size=2)
			# inShape is needed to pre-compute the indices for the unpooling (as they are not prouced by down pool)
			ind = np.mgrid[0 : inShape[0] * 2 : 2, 0 : inShape[1] * 2 : 2]
			ind_array = np.arange(inShape[0] * inShape[1] * 4).reshape((inShape[0] * 2, inShape[1] * 2))
			indices = np.zeros((dIn, inShape[0], inShape[1]), dtype=np.int32)
			indices[0 : ] = ind_array[ind[0], ind[1]]
			self.indices = indices.astype(np.int32)
			self.indicesShape = (dIn, inShape[0], inShape[1])
			# computeIndexes calls computeIndexesMethod to actually compute them
			self.computeIndexes = lambda x : self.computeIndexesMethod(x)
			# higher order function that calls the MaxUnpool2d layer.
			self.upSampleCall = lambda x, i : self.upSample(x, i)
		elif Type in ("bilinear", "nearest"):
			self.upSample = nn.Upsample(scale_factor=2, mode=Type)
			# computeIndexes return None as we don't need, just to avoid if in forward
			self.computeIndexes = lambda x : None
			# high order function that ignores 2nd param, just to avoid if in forward
			self.upSampleCall = lambda x, ind : self.upSample(x)
		elif Type == "conv_transposed":
			# (2, 2) x (3, 3) => (4, 4); (4, 4) x (5, 5) => (8, 8); (8, 8) x (9, 9) => (16, 16)
			self.upSample = nn.ConvTranspose2d(in_channels=dIn, out_channels=dIn, kernel_size=2, stride=2)
			self.computeIndexes = lambda x : None
			self.upSampleCall = lambda x, ind : F.relu(self.upSample(x, output_size=(2 * inShape[0], 2 * inShape[1])))

	# Fake indices for max unpooling, where it expects indices coming from a previous max pooling
	def computeIndexesMethod(self, x):
		# This cannot be pre-computed, due to variable mini batch size, so just copy the same indices over and over
		ind = np.zeros((x.data.shape[0], *self.indicesShape))
		ind[0 : ] = self.indices
		return Variable(maybeCuda(tr.from_numpy(ind).long()), requires_grad=False)

	def forward(self, x):
		ind = self.computeIndexes(x)
		y1 = self.upSampleCall(x, ind)
		y2 = F.relu(F.pad(self.conv(y1), (2, 2, 2, 2), "reflect"))
		return y2

class ModelLaina(NeuralNetworkPyTorch):
	def __init__(self, lossType, depthShape, baseModelType, upSampleType):
		super().__init__()
		assert baseModelType in ("resnet50", )
		assert upSampleType in ("unpool", "bilinear", "nearest", "conv_transposed")
		self.lossType = lossType
		self.depthShape = depthShape
		self.baseModelType = baseModelType
		self.upSampleType = upSampleType

		if self.lossType == "l2_loss":
			self.setCriterion(l2_loss)
		else:
			raise NotImplemented("Wrong loss type. Expected: l2_loss.")

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
		return "Laina. Base model: %s. Upsample type: %s. Loss Type: %s. Depth shape: %s" % (self.baseModelType, \
			self.upSampleType, self.lossType, self.depthShape)

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

		assert tuple(y_3_6.shape[1 : 3]) == self.depthShape
		return y_3_6