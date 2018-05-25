from datetime import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch as tr

from .utils import maybeCuda, maybeCpu
from .network import NeuralNetworkPyTorch, getNumParams, getOptimizerStr
from neural_wrappers.utilities import LinePrinter

class GenerativeAdversialNetwork(NeuralNetworkPyTorch):
	def __init__(self, generator, discriminator):
		super().__init__()
		self.generator = maybeCuda(generator)
		self.discriminator = maybeCuda(discriminator)

		self.metrics = {
			"LossG" : (lambda x, y, **k : k["lossG"]),
			"LossD" : (lambda x, y, **k : k["lossD"])
		}

	def summary(self):
		summaryStr = "[Model summary]\n"
		summaryStr += self.__str__() + "\n"

		gNumParams, gNumTrainable = getNumParams(self.generator.parameters())
		summaryStr += "Generator. Parameters count: %d. Trainable parameters: %d.\n" % (gNumParams, gNumTrainable)
		dNumParams, dNumTrainable = getNumParams(self.discriminator.parameters())
		summaryStr += "Discriminator. Parameters count: %d. Trainable parameters: %d.\n" % (dNumParams, dNumTrainable)

		strMetrics = str(list(self.metrics.keys()))[1 : -1]
		summaryStr += "Metrics: %s\n" % (strMetrics)

		summaryStr += "Optimizer: %s\n" % getOptimizerStr(self.optimizer)

		return summaryStr


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

	# @brief Runs one epoch of a GAN, using a data generator
	# @param[in] dataGenerator The generator from which data is retrieved. Only data (no labels) matter, but since
	#  the standard dataset reader API returns a tuple (data, labels), the second parameter is ignored.
	# @param[in] stepsPerEpoch How many steps the generator is expected to run
	# @param[in] callbacks A list of callbacks that are called. Only onEpochStart is called (not onIterationEnd yet)
	# @param[in] optimize A flag that specifies whether backpropagation is ran on the trainable weights or not.
	# @param[in] printMessage A flag that specifies whether the training iterations message is shown.
	# @param_optional[in] numStepsD Training GANs is tricky. Sometimes more successive steps are required for generator
	#  or discriminator, for better learning. This parameter specifies how many successive steps for discriminator are
	#  ran before training the generator in the alternative fashion. If not specified, defaults to 1.
	# @param_optional[in] numStepsG Same as numStepsD, but for generator.
	# @param_optional[in] optimizeD Whether to optimize the discriminator. If not specified, defaults to the value
	#  specified by the optimize parameter.
	# @param_optional[in] optimizeG Whether to optimize the generator. If not specified, defaults to the value
	#  specified by the optimize parameter.
	def run_one_epoch(self, dataGenerator, stepsPerEpoch, callbacks=[], optimize=False, \
		printMessage=False, **kwargs):
		if optimize:
			assert not self.generator.optimizer is None, "Set generator optimizer before training"
			assert not self.discriminator.optimizer is None, "Set discriminator optimizer before training"
		assert not self.criterion is None, "Set criterion before training or testing"
		assert "LossG" in self.metrics.keys(), "Generator Loss metric was not found in metrics."
		assert "LossD" in self.metrics.keys(), "Discriminator Loss metric was not found in metrics."

		numStepsD = kwargs["numStepsD"] if "numStepsD" in kwargs else 1
		numStepsG = kwargs["numStepsG"] if "numStepsG" in kwargs else 1
		assert numStepsD > 0 and numStepsG > 0
		optimizeD = kwargs["optimizeD"] if "optimizeD" in kwargs else optimize
		optimizeG = kwargs["optimizeG"] if "optimizeG" in kwargs else optimize

		self.checkCallbacks(callbacks)
		self.callbacksOnEpochStart(callbacks)

		metricResults = {metric : 0 for metric in self.metrics.keys()}
		i = 0

		self.discriminator.setTrainable(optimizeD)
		if optimizeD:
			optimizeCallbackD = (lambda optim, loss : (optim.zero_grad(), loss.backward(), optim.step()))
		else:
			optimizeCallbackD = (lambda optim, loss : loss.detach_())

		self.generator.setTrainable(optimizeG)
		if optimizeG:
			optimizeCallbackG = (lambda optim, loss : (optim.zero_grad(), loss.backward(), optim.step()))
		else:
			optimizeCallbackG = (lambda optim, loss : loss.detach_())

		startTime = datetime.now()
		while True:
			if i == stepsPerEpoch:
				break

			npLossD = 0
			actualNumStepsD = min(numStepsD, stepsPerEpoch - i)
			for j in range(actualNumStepsD):
				npInputs, _ = next(dataGenerator)
				trInputs = self.getTrData(npInputs, optimize=optimizeD)
				batchSize = npInputs.shape[0]

				# Adversarial ground truths
				fakeLabels = Variable(maybeCuda(tr.zeros(batchSize)))
				validLabels = Variable(maybeCuda(tr.ones(batchSize)))

				# Train Discriminator
				# Measure discriminator's ability to classify real from generated samples
				z = Variable(maybeCuda(tr.randn(batchSize, self.generator.inputSize)))
				trGeneratedInput = self.generator(z)
				trRealLossD = self.criterion(self.discriminator(trInputs), validLabels)
				trFakeLossD = self.criterion(self.discriminator(trGeneratedInput.detach()), fakeLabels)
				trLossD = (trRealLossD + trFakeLossD) / 2
				npLossD += maybeCpu(trLossD.data).numpy()
				optimizeCallbackD(self.discriminator.optimizer, trLossD)

			npLossG = 0
			for j in range(numStepsG):
				# Train Generator
				# Sample noise as generator input and generate a bunch of data
				z = Variable(maybeCuda(tr.randn(batchSize, self.generator.inputSize)))
				trGeneratedInput = self.generator(z)
				# Loss measures generator's ability to fool the discriminator
				trLossG = self.criterion(self.discriminator(trGeneratedInput), validLabels)
				npLossG += maybeCpu(trLossG.data).numpy()
				optimizeCallbackG(self.generator.optimizer, trLossG)

			# TODO: see what kind of inputs, may be good here, for now keep API same for metrics.
			for metric in self.metrics:
				metricResults[metric] += self.metrics[metric](None, None, lossG=npLossG, lossD=npLossD)

			iterFinishTime = (datetime.now() - startTime)
			if printMessage:
				self.linePrinter.print(self.computeIterPrintMessage(i, stepsPerEpoch, metricResults, iterFinishTime))
			i += actualNumStepsD

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