import torch as tr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_wrapper import NeuralNetworkPyTorch, maybeCuda

# UnPool Layer as defined by Laina paper
class UpSampleUnpool(NeuralNetworkPyTorch):
	def __init__(self, inShape, dIn, dOut):
		super(UpSampleUnpool, self).__init__()
		assert not inShape is None
		self.upSample = nn.MaxUnpool2d(kernel_size=2)
		# inShape is needed to pre-compute the indices for the unpooling (as they are not prouced by down pool)
		ind = np.mgrid[0 : inShape[0] * 2 : 2, 0 : inShape[1] * 2 : 2]
		ind_array = np.arange(inShape[0] * inShape[1] * 4).reshape((inShape[0] * 2, inShape[1] * 2))
		indices = np.zeros((dIn, inShape[0], inShape[1]), dtype=np.int32)
		indices[0 : ] = ind_array[ind[0], ind[1]]
		self.indices = indices.astype(np.int32)
		self.indicesShape = (dIn, inShape[0], inShape[1])

	# Fake indices for max unpooling, where it expects indices coming from a previous max pooling
	def computeIndexesMethod(self, x):
		# This cannot be pre-computed, due to variable mini batch size, so just copy the same indices over and over
		ind = np.zeros((x.data.shape[0], *self.indicesShape))
		ind[0 : ] = self.indices
		trInd = tr.from_numpy(ind).long()

		return Variable(maybeCuda(trInd), requires_grad=False)

	def forward(self, x):
		ind = self.computeIndexesMethod(x)
		return self.upSample(x, ind)

class UpSampleConvTransposed(NeuralNetworkPyTorch):
	def __init__(self, dIn, dOut):
		super(UpSampleConvTransposed, self).__init__()
		self.upSample = nn.ConvTranspose2d(in_channels=dIn, out_channels=dIn, kernel_size=2, stride=2)

	def forward(self, x):
		out = F.relu(self.upSample(x))
		# MB x dIn x H x W => MB x dOut x 2*H x 2*W
		assert x.shape[2] * 2 == out.shape[2] and x.shape[3] * 2 == out.shape[3], "Expected upconv transposed to " + \
			"double the output shape, got: in %s, out %s" % (x.shape[2 : 4], out.shape[2 : 4])
		return out

# Class that implements 3 methods for up-sampling from the bottom encoding layer
# "unpool" is the method described in Laina paper, with unpooling method with zeros + conv
# "nearest" and "bilinear" are based on using PyTorch's nn.Upsample layer
class UpSampleLayer(NeuralNetworkPyTorch):
	def __init__(self, inShape, dIn, dOut, Type):
		super(UpSampleLayer, self).__init__()
		assert Type in ("unpool", "bilinear", "nearest", "conv_transposed")
		assert (Type != "unpool") or (Type == "unpool" and len(inShape) == 2), "For unpool, inShape must be provided \
			as a tuple so indices for max unpool can be pre-computed."
		self.conv = nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=5)

		if Type == "unpool":
			self.upSampleLayer = UpSampleUnpool(inShape=inShape, dIn=dIn, dOut=dOut)
		elif Type in ("bilinear", "nearest"):
			self.upSampleLayer = nn.Upsample(scale_factor=2, mode=Type)
		elif Type == "conv_transposed":
			self.upSampleLayer = UpSampleConvTransposed(dIn=dIn, dOut=dOut)

	def forward(self, x):
		y1 = self.upSampleLayer(x)
		y2 = F.relu(F.pad(self.conv(y1), (2, 2, 2, 2), "reflect"))
		return y2
