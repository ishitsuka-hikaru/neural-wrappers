from neural_wrappers.pytorch import NeuralNetworkPyTorch
from .model_unet import UNetBlock
import torch.nn as nn
import torch.nn.functional as F
import torch as tr

class ConcatenateBlock(NeuralNetworkPyTorch):
	def __init__(self, dIn, dOut):
		super().__init__()
		self.convt = nn.ConvTranspose2d(in_channels=dIn, out_channels=dOut, kernel_size=3, stride=(2, 2))

	def forward(self, x_down, x_up):
		y_up = F.relu(self.convt(x_up))
		y_up = F.pad(y_up, (0, -1, 0, -1))
		y_concat = tr.cat([x_down, y_up], dim=1)
		return y_concat

class ModelUNetDilatedConv(NeuralNetworkPyTorch):
	def __init__(self, inputShape, numFilters):
		super().__init__()

		# Feature extractor part (down)
		inChannels = inputShape[-1]
		self.downBlock1 = UNetBlock(dIn=inChannels, dOut=numFilters, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2)
		self.downBlock2 = UNetBlock(dIn=numFilters, dOut=numFilters * 2, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2)
		self.downBlock3 = UNetBlock(dIn=numFilters * 2, dOut=numFilters * 4, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2)

		# Stacked dilated convolution part
		self.dilate1 = nn.Conv2d(in_channels=numFilters * 4, out_channels=numFilters * 8, \
			kernel_size=3, padding=(1, 1), dilation=1)
		self.dilate2 = nn.Conv2d(in_channels=numFilters * 8, out_channels=numFilters * 8, \
			kernel_size=3, padding=(2, 2), dilation=2)
		self.dilate3 = nn.Conv2d(in_channels=numFilters * 8, out_channels=numFilters * 8, \
			kernel_size=3, padding=(3, 3), dilation=4)
		self.dilate4 = nn.Conv2d(in_channels=numFilters * 8, out_channels=numFilters * 8, \
			kernel_size=3, padding=(4, 4), dilation=8)
		self.dilate5 = nn.Conv2d(in_channels=numFilters * 8, out_channels=numFilters * 8, \
			kernel_size=3, padding=(5, 5), dilation=16)
		self.dilate6 = nn.Conv2d(in_channels=numFilters * 8, out_channels=numFilters * 8, \
			kernel_size=3, padding=(6, 6), dilation=32)

		# Final up-sample layers
		# Input to up3 is the output of the concatenated dilated convs (6 concatenations of numFilters * 8)
		self.up3 = ConcatenateBlock(dIn=numFilters * 8 * 6, dOut=numFilters * 4)
		self.upBlock3 = UNetBlock(dIn=numFilters * 8, dOut=numFilters * 4, padding=1)
		self.up2 = ConcatenateBlock(dIn=numFilters * 4, dOut=numFilters * 2)
		self.upBlock2 = UNetBlock(dIn=numFilters * 4, dOut=numFilters * 2, padding=1)
		self.up1 = ConcatenateBlock(dIn=numFilters * 2, dOut=numFilters)
		self.upBlock1 = UNetBlock(dIn=numFilters * 2, dOut=numFilters, padding=1)

		self.finalConv = nn.Conv2d(in_channels=numFilters, out_channels=1, kernel_size=(1, 1))

	def forward(self, x):
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)
		y_down1 = self.downBlock1(x)
		y_down1pool = self.pool1(y_down1)
		y_down2 = self.downBlock2(y_down1pool)
		y_down2pool = self.pool2(y_down2)
		y_down3 = self.downBlock3(y_down2pool)
		y_down3pool = self.pool3(y_down3)

		y_dilate1 = F.relu(self.dilate1(y_down3pool))
		y_dilate2 = F.relu(self.dilate2(y_dilate1))
		y_dilate3 = F.relu(self.dilate2(y_dilate2))
		y_dilate4 = F.relu(self.dilate2(y_dilate3))
		y_dilate5 = F.relu(self.dilate2(y_dilate4))
		y_dilate6 = F.relu(self.dilate2(y_dilate5))
		y_dilate_concatenate = tr.cat([y_dilate1, y_dilate2, y_dilate3, y_dilate4, y_dilate5, y_dilate6], dim=1)

		y_up3 = self.up3(y_down3, y_dilate_concatenate)
		y_up3block = self.upBlock3(y_up3)
		y_up2 = self.up2(y_down2, y_up3block)
		y_up2block = self.upBlock2(y_up2)
		y_up1 = self.up1(y_down1, y_up2block)
		y_up1block = self.upBlock1(y_up1)

		y_final = self.finalConv(y_up1block)
		y_final = y_final.view(y_final.shape[0], y_final.shape[2], y_final.shape[3])
		return y_final