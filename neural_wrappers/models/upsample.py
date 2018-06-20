import torch as tr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda

# UnPool Layer as defined by Laina paper
class UpSampleUnpool(NeuralNetworkPyTorch):
	# @param[in] dIn Number of depth channels for the input
	# @param[in] inShape  If inShape is provided, then the indices are pre-computed and will be used at every forward
	#  step. Otherwise, if they are not provided (or set to None), they will be computed at every forward step.
	def __init__(self, dIn, inShape=None):
		super(UpSampleUnpool, self).__init__()
		self.upSample = nn.MaxUnpool2d(kernel_size=2)
		self.dIn = dIn
		if inShape != None:
			self.preComputedIndices = self.computeIndices(inShape)
		else:
			self.preComputedIndices = None

	def computeIndices(self, inShape):
		# inShape is needed to pre-compute the indices for the unpooling (as they are not prouced by down pool)
		ind = np.mgrid[0 : inShape[0] * 2 : 2, 0 : inShape[1] * 2 : 2]
		ind_array = np.arange(inShape[0] * inShape[1] * 4).reshape((inShape[0] * 2, inShape[1] * 2))
		indices = np.zeros((self.dIn, inShape[0], inShape[1]), dtype=np.int32)
		indices[0 : ] = ind_array[ind[0], ind[1]]
		return np.int32(indices)

	# Fake indices for max unpooling, where it expects indices coming from a previous max pooling
	def getIndices(self, x):
		assert x.shape[1] == self.dIn
		if self.preComputedIndices is None:
			indices = self.computeIndices((x.shape[2], x.shape[3]))
		else:
			indices = self.preComputedIndices

		# This cannot be pre-computed, due to variable mini batch size, so just copy the same indices over and over
		miniBatchSize = x.data.shape[0]
		miniBatchIndices = np.zeros((x.data.shape[0], *indices.shape))
		miniBatchIndices[0 : ] = indices
		trInd = tr.from_numpy(miniBatchIndices).long()
		return Variable(maybeCuda(trInd), requires_grad=False)

	def forward(self, x):
		ind = self.getIndices(x)
		return self.upSample(x, ind)

class UpSampleConvTransposed(NeuralNetworkPyTorch):
	def __init__(self, dIn):
		super(UpSampleConvTransposed, self).__init__()
		self.upSample = nn.ConvTranspose2d(in_channels=dIn, out_channels=dIn, kernel_size=2, stride=2)

	def forward(self, x):
		out = F.relu(self.upSample(x))
		# MB x dIn x H x W => MB x dIn x 2*H x 2*W
		assert x.shape[2] * 2 == out.shape[2] and x.shape[3] * 2 == out.shape[3], "Expected upconv transposed to " + \
			"double the output shape, got: in %s, out %s" % (x.shape[2 : 4], out.shape[2 : 4])
		return out

# Class that implements 3 methods for up-sampling from the bottom encoding layer
# "unpool" is the method described in Laina paper, with unpooling method with zeros + conv
# "nearest" and "bilinear" are based on using PyTorch's nn.Upsample layer
class UpSampleLayer(NeuralNetworkPyTorch):
	# @param[in] dIn Number of depth channels for input
	# @param[in] dOut number of depth channels for output (last convolution uses this parameter)
	# @param[in] Type The type of upsampling method. Valid values: "unpool", "nearest", "bilinear" or "conv_transposed"
	# @optional_param[in] inShape Used by unpool method to pre-compute indices for unpooling. If not provided, they are
	#  computed dynamically during each forward phase based on the input shape.
	def __init__(self, dIn, dOut, Type, **kwargs):
		super(UpSampleLayer, self).__init__()
		assert Type in ("unpool", "bilinear", "nearest", "conv_transposed")
		self.conv = nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=5)

		if Type == "unpool":
			inShape = kwargs["inShape"] if "inShape" in kwargs else None
			self.upSampleLayer = UpSampleUnpool(dIn=dIn, inShape=inShape)
		elif Type in ("bilinear", "nearest"):
			self.upSampleLayer = nn.Upsample(scale_factor=2, mode=Type, align_corners=True)
		elif Type == "conv_transposed":
			self.upSampleLayer = UpSampleConvTransposed(dIn=dIn)

	def forward(self, x):
		y1 = self.upSampleLayer(x)
		y2 = F.relu(F.pad(self.conv(y1), (2, 2, 2, 2), "reflect"))
		return y2
