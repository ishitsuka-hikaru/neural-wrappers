import torch as tr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_wrapper import maybeCuda

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
