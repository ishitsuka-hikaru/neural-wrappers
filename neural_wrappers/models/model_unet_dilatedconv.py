import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from .upsample import UpSampleLayer
from .model_unet import UNetBlock
from ..pytorch import FeedForwardNetwork

class ConcatenateBlock(FeedForwardNetwork):
	def __init__(self, dIn, dOut):
		super().__init__()
		self.convt = UpSampleLayer(dIn=dIn, dOut=dOut, Type="conv_transposed", noSmoothing=True, \
			convTransposedKernelSize=3, convTransposedStride=2)

	def forward(self, x_down, x_up):
		y_up = F.relu(self.convt(x_up))
		y_up = F.pad(y_up, (0, -1, 0, -1))
		y_concat = tr.cat([x_down, y_up], dim=1)
		return y_concat

# Class that computes the layers in the bottleneck portion of the model
# TODO: further generalize instead of using "modes"
class BottleneckBlock(FeedForwardNetwork):
	def __init__(self, dIn, mode, numFilters=6):
		super().__init__()
		assert mode in ("dilate2_serial_concatenate", "dilate2_parallel_concatenate", "dilate2_serial_sum")
		self.dIn = dIn
		self.mode = mode
		self.numFilters = numFilters
		self.dilateLayers = nn.ModuleList()

		# First convolution receives dIn dimensions, and outputs dIn* 2, the rest (because it's serial), filter
		#  the dIn * 2 dimensions, into other seris of dIn * 2
		if mode == "dilate2_serial_concatenate":
			for i in range(numFilters):
				in_channels = dIn if i == 0 else dIn * 2
				newLayer = nn.Conv2d(in_channels=in_channels, out_channels=dIn * 2, kernel_size=3, \
					padding=2**i, dilation=2**i)
				self.dilateLayers.append(newLayer)
			# The last layer concatentes all the numFilters features, which have a shape of dIn * 2 dimensions
			self.finalLayer = ConcatenateBlock(dIn=dIn * 2 * numFilters, dOut=dIn)
			self.forwardMethod = self.dilate2_serial_concatenate
		# All the convolutions are applied to the first feature map (of shape dIn), then all of them are concatenated
		#  at the end.
		elif mode == "dilate2_parallel_concatenate":
			for i in range(numFilters):
				newLayer = nn.Conv2d(in_channels=dIn, out_channels=dIn * 2, kernel_size=3, \
					padding=2**i, dilation=2**i)
				self.dilateLayers.append(newLayer)
			# The last layer concatentes all the numFilters features, which have a shape of dIn * 2 dimensions
			self.finalLayer = ConcatenateBlock(dIn=dIn * 2 * numFilters, dOut=dIn)
			self.forwardMethod = self.dilate2_parallel_concatenate

	def dilate2_serial_concatenate(self, x_down, x_pooled):
		y = x_pooled
		concatenatedFeatures = []
		for i in range(self.numFilters):
			y = F.relu(self.dilateLayers[i](y))
			concatenatedFeatures.append(y)
		yDilateConcatenate = tr.cat(concatenatedFeatures, dim=1)
		# Final portion concatenates the features with the resdiual connection
		yFinal = self.finalLayer(x_down, yDilateConcatenate)
		return yFinal

	def dilate2_parallel_concatenate(self, x_down, x_pooled):
		concatenatedFeatures = []
		for i in range(self.numFilters):
			y = F.relu(self.dilateLayers[i](x_pooled))
			concatenatedFeatures.append(y)
		yDilateConcatenate = tr.cat(concatenatedFeatures, dim=1)
		yFinal = self.finalLayer(x_down, yDilateConcatenate)
		return yFinal

	def forward(self, x_down, x_pooled):
		return self.forwardMethod(x_down, x_pooled)

class ModelUNetDilatedConv(FeedForwardNetwork):
	def __init__(self, dIn, dOut, numFilters, bottleneckMode):
		super().__init__()

		# Feature extractor part (down)
		self.downBlock1 = UNetBlock(dIn=dIn, dOut=numFilters, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2)
		self.downBlock2 = UNetBlock(dIn=numFilters, dOut=numFilters * 2, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2)
		self.downBlock3 = UNetBlock(dIn=numFilters * 2, dOut=numFilters * 4, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2)

		self.bottleneck = BottleneckBlock(numFilters * 4, mode=bottleneckMode, numFilters=6)

		# Final up-sample layers
		self.upBlock3 = UNetBlock(dIn=numFilters * 8, dOut=numFilters * 4, padding=1)
		self.up2 = ConcatenateBlock(dIn=numFilters * 4, dOut=numFilters * 2)
		self.upBlock2 = UNetBlock(dIn=numFilters * 4, dOut=numFilters * 2, padding=1)
		self.up1 = ConcatenateBlock(dIn=numFilters * 2, dOut=numFilters)
		self.upBlock1 = UNetBlock(dIn=numFilters * 2, dOut=numFilters, padding=1)

		self.finalConv = nn.Conv2d(in_channels=numFilters, out_channels=dOut, kernel_size=(1, 1))

	def forward(self, x):
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)
		y_down1 = self.downBlock1(x)
		y_down1pool = self.pool1(y_down1)
		y_down2 = self.downBlock2(y_down1pool)
		y_down2pool = self.pool2(y_down2)
		y_down3 = self.downBlock3(y_down2pool)
		y_down3pool = self.pool3(y_down3)

		y_bottleneck = self.bottleneck(y_down3, y_down3pool)

		y_up3block = self.upBlock3(y_bottleneck)
		y_up2 = self.up2(y_down2, y_up3block)
		y_up2block = self.upBlock2(y_up2)
		y_up1 = self.up1(y_down1, y_up2block)
		y_up1block = self.upBlock1(y_up1)

		y_final = self.finalConv(y_up1block)
		y_final = y_final.transpose(1, 3).transpose(1, 2)
		return y_final