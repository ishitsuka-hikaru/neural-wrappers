import sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from overrides import overrides

from .nw_module import NWModule
from .utils import device
from ..utilities import isType

def lossFn(y, t):
	return -(t * tr.log(y + 1e-5) + (1 - t) * tr.log(1 - y + 1e-5)).mean()

class GANOptimizer(optim.Optimizer):
	def __init__(self, model, optimizer, **kwargs):
		self.model = model
		model.generator.setOptimizer(optimizer, **kwargs)
		model.discriminator.setOptimizer(optimizer, **kwargs)
		self.param_groups = model.generator.getOptimizer().param_groups + \
			model.discriminator.getOptimizer().param_groups

	def state_dict(self):
		return {
			"discriminator" : self.model.discriminator.getOptimizer(),
			"generator" : self.model.generator.getOptimizer()
		}

	def load_state_dict(self, state):
		self.model.discriminator.getOptimizer().load_state_dict(state["discriminator"])
		self.model.generator.getOptimizer().load_state_dict(state["generator"])

	@overrides
	def step(self, closure=None):
		self.model.discriminator.getOptimizer().step(closure)
		self.model.generator.getOptimizer().step(closure)

	@overrides
	def __str__(self):
		Str = "[Gan Optimizer]"
		Str += "\n - Generator: %s" % self.model.generator.getOptimizerStr()
		Str += "\n - Discriminator: %s" % self.model.discriminator.getOptimizerStr()
		return Str

	def __getattr__(self, key):
		assert key in ("discriminator", "generator")
		return self.state_dict()[key]

class GenerativeAdversarialNetwork(NWModule):
	def __init__(self, generator:NWModule, discriminator:NWModule):
		super().__init__()
		assert hasattr(generator, "noiseSize")
		self.generator = generator
		self.discriminator = discriminator
		# self.discriminator.setCriterion(lossFn)
		self.setCriterion(lossFn)

	@overrides
	def setOptimizer(self, optimizer, **kwargs):
		assert not isinstance(optimizer, optim.Optimizer)
		ganOptimizer = GANOptimizer(self, optimizer, **kwargs)
		super().setOptimizer(ganOptimizer, **kwargs)

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

		lossDReal = self.criterion(predictDReal, ones)
		lossDFake = self.criterion(predictDFake, zeros)
		lossD = lossDReal + lossDFake
		if isTraining and isOptimizing:
			self.getOptimizer().optimizer["discriminator"].zero_grad()
			lossD.backward()
			self.getOptimizer().optimizer["discriminator"].step()
		else:
			lossD.detach_()

		# Train generator
		predictGFake = self.discriminator.forward(fakeG)
		lossG = self.criterion(predictGFake, ones)
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
