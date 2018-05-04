from datetime import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch as tr

from .utils import maybeCuda, maybeCpu
from .network import NeuralNetworkPyTorch
from neural_wrappers.utilities import LinePrinter

class GenerativeAdversialNetwork(NeuralNetworkPyTorch):
	def __init__(self, generator, discriminator):
		super().__init__()
		self.generator = maybeCuda(generator)
		self.discriminator = maybeCuda(discriminator)
		self.linePrinter = LinePrinter()

		self.metrics = {
			"LossG" : (lambda x, y, **k : k["lossG"]),
			"LossD" : (lambda x, y, **k : k["lossD"])
		}

	def setMetrics(self, metrics):
		assert not "LossG" in metrics, "Cannot overwrite generator Loss metric."
		assert not "LossD" in metrics, "Cannot overwrite discriminator Loss metric."

		for key in metrics:
			assert type(key) == str, "The key of the metric must be a string"
			assert hasattr(metrics[key], "__call__"), "The user provided transformation %s must be callable" % (key)
		self.metrics = metrics
		# Set LossG and LossD by default
		self.metrics["LossG"] = lambda x, y, **k : k["lossG"]
		self.metrics["LossD"] = lambda x, y, **k : k["lossD"]

	def run_one_epoch(self, generator, stepsPerEpoch, callbacks=[], optimize=False, printMessage=False):
		if optimize:
			assert not self.generator.optimizer is None, "Set generator optimizer before training"
			assert not self.discriminator.optimizer is None, "Set discriminator optimizer before training"
		assert not self.criterion is None, "Set criterion before training or testing"
		assert "LossG" in self.metrics.keys(), "Generator Loss metric was not found in metrics."
		assert "LossD" in self.metrics.keys(), "Discriminator Loss metric was not found in metrics."
		self.checkCallbacks(callbacks)
		self.callbacksOnEpochStart(callbacks)

		metricResults = {metric : 0 for metric in self.metrics.keys()}
		linePrinter = LinePrinter()
		i = 0

		startTime = datetime.now()
		for i, items in enumerate(generator):
			npInputs, _ = items
			trInputs = self.getTrData(npInputs, optimize=optimize)
			batchSize = npInputs.shape[0]

			# Adversarial ground truths
			fakeLabels = Variable(maybeCuda(tr.zeros(batchSize)))
			validLabels = Variable(maybeCuda(tr.ones(batchSize)))

			# Train Generator
			# Sample noise as generator input and generate a bunch of data
			z = Variable(maybeCuda(tr.randn(batchSize, self.generator.inputSize)))
			trGeneratedInput = self.generator(z)

			# Loss measures generator's ability to fool the discriminator
			trLossG = self.criterion(self.discriminator(trGeneratedInput), validLabels)
			npLossG = maybeCpu(trLossG.data).numpy()

			if optimize:
				self.generator.optimizer.zero_grad()
				trLossG.backward()
				self.generator.optimizer.step()

			# Train Discriminator
			# Measure discriminator's ability to classify real from generated samples
			trRealLossD = self.criterion(self.discriminator(trInputs), validLabels)
			trFakeLossD = self.criterion(self.discriminator(trGeneratedInput.detach()), fakeLabels)
			trLossD = (trRealLossD + trFakeLossD) / 2
			npLossD = maybeCpu(trLossD.data).numpy()

			if optimize:
				self.discriminator.optimizer.zero_grad()
				trLossD.backward()
				self.discriminator.optimizer.step()

			# TODO: see what kind of inputs, may be good here, for now keep API same for metrics.
			for metric in self.metrics:
				metricResults[metric] += self.metrics[metric](None, None, lossG=npLossG, lossD=npLossD)

			iterFinishTime = (datetime.now() - startTime)
			if printMessage:
				linePrinter.print(self.computeIterPrintMessage(i, stepsPerEpoch, metricResults, iterFinishTime))

			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			sys.stderr.write("Warning! Number of iterations (%d) does not match expected iterations in reader (%d)" % \
				(i, stepsPerEpoch - 1))
		for metric in metricResults:
			metricResults[metric] /= stepsPerEpoch
		return metricResults

	def save_model(self, path):
		generatorState = {
			"weights" : list(map(lambda x : x.cpu(), self.generator.parameters())),
			"optimizer_type" : type(self.generator.optimizer),
			"optimizer_state" : self.generator.optimizer.state_dict(),
		}

		discriminatorState = {
			"weights" : list(map(lambda x : x.cpu(), self.discriminator.parameters())),
			"optimizer_type" : type(self.discriminator.optimizer),
			"optimizer_state" : self.discriminator.optimizer.state_dict(),
		}

		state = {
			"generatorState" : generatorState,
			"discriminatorState" : discriminatorState,
			"history_dict" : self.trainHistory
		}

		tr.save(state, path)

	def load_model(self, path):
		loaded_model = tr.load(path)
		generatorState = loaded_model["generatorState"]
		discriminatorState = loaded_model["discriminatorState"]
		self.generator._load_model(generatorState)
		self.discriminator._load_model(discriminatorState)

		if "history_dict" in loaded_model:
			self.load_history(loaded_model["history_dict"])
			print("Succesfully loaded GAN model (with history, epoch %d)" % (self.currentEpoch))
		else:
			print("Succesfully loaded GAN model (no history)")