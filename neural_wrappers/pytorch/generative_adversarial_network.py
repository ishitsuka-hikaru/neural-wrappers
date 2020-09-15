import sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from overrides import overrides

from .nw_module import NWModule
from .utils import device
from ..utilities import MessagePrinter, RunningMean, isType

def lossFn(y, t):
	return -(t * tr.log(y + 1e-5) + (1 - t) * tr.log(1 - y + 1e-5)).mean()

def f(y, t):
	breakpoint()

class GANOptimizer(optim.Optimizer):
	def __init__(self, optimizerDict):
		assert "generator" in optimizerDict
		assert "discriminator" in optimizerDict
		self.optimizer = optimizerDict

	def state_dict(self):
		return {
			"generator" : self.optimizer["generator"].state_dict(),
			"discriminator" : self.optimizer["discriminator"].state_dict()
		}

	def load_state_dict(self, state):
		for k in ["generator", "discriminator"]:
			self.optimizer[k].load_state_dict(state[k])

	def __str__(self):
		return "GAN OPTIMIZER"

class GenerativeAdversarialNetwork(NWModule):
	def __init__(self, generator:NWModule, discriminator:NWModule):
		super().__init__()
		assert hasattr(generator, "noiseSize")
		self.generator = generator
		self.discriminator = discriminator
		self.discriminator.setCriterion(lossFn)

		self.setCriterion(f)

	@overrides
	def setOptimizer(self, optimizer, **kwargs):
		optimizerDict = {
			"generator" : optimizer(self.generator.parameters(), **kwargs),
			"discriminator" : optimizer(self.discriminator.parameters(), **kwargs)
		}

		optimizer = GANOptimizer(optimizerDict)
		super().setOptimizer(optimizer)

		# if isType(optimizer, dict):
		# 	Keys = list(optimizer.keys())
		# 	assert len(Keys) == 2
		# 	assert "generator" in Keys
		# 	assert "discriminator" in Keys
		# 	assert "generator" in kwargs
		# 	assert "discriminator" in kwargs


		# 	# self.generator.setOptimizer(optimizer["generator"], **kwargs["generator"])
		# 	# self.discriminator.setOptimizer(optimizer["discriminator"], **kwargs["discriminator"])
		# else:
		# 	self.generator.setOptimizer(optimizer, **kwargs)
		# 	self.discriminator.setOptimizer(optimizer, **kwargs)

		# self.optimizer = {
		# 	"generator" : self.generator.getOptimizer(),
		# 	"discriminator" : self.discriminator.getOptimizer()
		# }
		# self.optimizer.storedArgs = kwargs

	def updateOptimizer(self, trLoss, isTraining, isOptimizing, retain_graph=False):
		if not trLoss is None:
			if isTraining and isOptimizing:
				self.getOptimizer().zero_grad()
				trLoss.backward(retain_graph=retain_graph)
				self.getOptimizer().step()
			else:
				trLoss.detach_()

	@overrides
	def networkAlgorithm(self, trInputs, trLabels, isTraining, isOptimizing):
		MB = len(trInputs)
		ones = tr.full((MB, ), 1, dtype=tr.float32).to(device)
		zeros = tr.full((MB, ), 0, dtype=tr.float32).to(device)
		noiseD = tr.randn((MB, self.generator.noiseSize)).to(device)
		noiseG = tr.randn((MB, self.generator.noiseSize)).to(device)

		# Detach the fake data generated for discriminator from generator, so they are trained independently
		fakeD = self.generator.forward(noiseD).detach()
		fakeG = self.generator.forward(noiseG)

		# Train discriminator
		predictDReal = self.discriminator.forward(trInputs)
		predictDFake = self.discriminator.forward(fakeD)

		lossDReal = self.discriminator.criterion(predictDReal, ones)
		lossDFake = self.discriminator.criterion(predictDFake, zeros)
		lossD = lossDReal + lossDFake
		if isTraining and isOptimizing:
			self.getOptimizer().optimizer["discriminator"].zero_grad()
			lossD.backward()
			self.getOptimizer().optimizer["discriminator"].step()
		else:
			lossD.detach_()

		# Train generator
		predictGFake = self.discriminator.forward(fakeG)
		lossG = self.discriminator.criterion(predictGFake, ones)
		if isTraining and isOptimizing:
			self.getOptimizer().optimizer["generator"].zero_grad()
			lossG.backward()
			self.getOptimizer().optimizer["generator"].step()
		else:
			lossG.detach_()

		trResults = {
			"generator" : {"fakeD" : fakeD, "fakeG" : fakeG},
			"discriminator" : {"predictDReal" : predictDReal, "predictDFake" : predictDFake, \
				"predictGFake" : predictGFake}
		}
		# trLoss = {
		# 	"generator" : lossG,
		# 	"discriminator" : {"lossDReal" : lossDReal, "lossDFake" : lossDFake}
		# }
		# These two should sum to 0.5 in theory when it stabilizes (discriminator is fully fooled)
		trLoss = (lossG + lossDFake) / 2
		return trResults, trLoss
