import sys
import torch.nn as nn
import torch as tr
import numpy as np
import torch.optim as optim

from .network import NeuralNetworkPyTorch
from neural_wrappers.utilities import MessagePrinter, RunningMean
from datetime import datetime
from functools import partial

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

def makeGenerator(data, labels):
	yield data, labels

def lossFn(y, t):
	return -(t * tr.log(y + 1e-5) + (1 - t) * tr.log(1 - y + 1e-5)).mean()

class GenerativeAdversarialNetwork(NeuralNetworkPyTorch):
	def __init__(self, generator, discriminator):
		super().__init__()
		self.generator = generator
		self.discriminator = discriminator

		assert self.discriminator.criterion is None
		# Add a dummy criterion if none is set so we can train
		if self.generator.criterion is None:
			self.generator.setCriterion(lambda y, t : tr.FloatTensor([0]).to(device).requires_grad_(True))
		self.generator.networkAlgorithm = partial(GenerativeAdversarialNetwork.generatorNetworkAlgorithm, \
			generator=self.generator, discriminator=self.discriminator)
		self.discriminator.setCriterion(lossFn)
		self.setOptimizer(optim.SGD, lr=0.01)

		self.iterPrintMessageKeys = self.generator.iterPrintMessageKeys
		for key in self.discriminator.iterPrintMessageKeys:
			self.iterPrintMessageKeys.append("%s (D)" % (key))

	# Kinda obtuse implementation. Basically call generator's criterion as well as discriminator's criterion, but
	#  without optimizing the discriminator
	def generatorNetworkAlgorithm(trInputs, trLabels, generator, discriminator):
		yGenerator = generator(trInputs)
		generatorLoss = generator.criterion(yGenerator, trLabels)
		discriminator.requires_grad_(False)
		yDiscriminator = discriminator(yGenerator)
		MB = trInputs.shape[0]
		ones = tr.ones(MB).to(device)
		discriminatorLoss = discriminator.criterion(yDiscriminator, ones)
		discriminator.requires_grad_(True)

		return yGenerator, discriminatorLoss + generatorLoss

	def callbacksOnEpochStart(self, isTraining):
		self.generator.callbacksOnEpochStart(isTraining)
		self.discriminator.callbacksOnEpochStart(isTraining)
		super().callbacksOnEpochStart(isTraining)

	def callbacksOnEpochEnd(self, isTraining):
		self.generator.callbacksOnEpochEnd(isTraining)
		self.discriminator.callbacksOnEpochEnd(isTraining)
		super().callbacksOnEpochEnd(isTraining)

	def getCombinedMetrics(self, metrics):
		if metrics is None:
			return None

		combinedMetricResults = {k : metrics["generator"][k] \
			for k in filter(lambda x : x in self.generator.iterPrintMessageKeys , metrics["generator"].keys())}
		for k in metrics["discriminator"]:
			if not k in self.discriminator.iterPrintMessageKeys:
				continue
			combinedMetricResults["%s (D)" % (k)] = metrics["discriminator"][k]
		return combinedMetricResults

	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		combinedTrainMetrics = self.getCombinedMetrics(trainMetrics)
		combinedValidationMetrics = self.getCombinedMetrics(validationMetrics)
		return super().computePrintMessage(combinedTrainMetrics, combinedValidationMetrics, numEpochs, duration)

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		combinedMetricResults = self.getCombinedMetrics(metricResults)
		return super().computeIterPrintMessage(i, stepsPerEpoch, combinedMetricResults, iterFinishTime)

	def summary(self):
		summaryStr = "[GAN summary]\n"
		summaryStr += self.__str__() + "\n"

		summaryStr += "[Generator]\n"
		summaryStr += self.generator.summary() + "\n\n"
		summaryStr += "[Discriminator]\n"
		summaryStr += self.discriminator.summary() + "\n"

		return summaryStr

	def epochPrologue(self, epochMetrics):
		self.linePrinter(epochMetrics["message"])

		generatorEpochMetrics = {"Train" : epochMetrics["Train"]["generator"], "message" : epochMetrics["message"], \
			"duration" : epochMetrics["duration"]}
		discriminatorEpochMetrics = {"Train" : epochMetrics["Train"]["generator"], \
			"message" : epochMetrics["message"], "duration" : epochMetrics["duration"]}
		ganEpochMetrics = {
			"Train" : self.getCombinedMetrics({"generator" : generatorEpochMetrics["Train"], \
				"discriminator" : discriminatorEpochMetrics["Train"]
			}),
			"duration" : epochMetrics["duration"],
			"message" : epochMetrics["message"]
		}
		if "Validation" in epochMetrics:
			generatorEpochMetrics["Validation"] = epochMetrics["Validation"]["generator"]
			discriminatorEpochMetrics["Validation"] = epochMetrics["Validation"]["discriminator"]
			ganEpochMetrics["Validation"] = self.getCombinedMetrics({
				"generator" : generatorEpochMetrics["Validation"], \
				"discriminator" : discriminatorEpochMetrics["Validation"]
			})

		self.generator.epochPrologue(generatorEpochMetrics)
		self.discriminator.epochPrologue(discriminatorEpochMetrics)

		# Combine Train/val from generator/discriminator
		self.trainHistory.append(ganEpochMetrics)
		self.callbacksOnEpochEnd(isTraining=True)
		if not self.optimizerScheduler is None:
			self.optimizerScheduler.step()
		self.currentEpoch += 1

	def run_one_epoch(self, generator, stepsPerEpoch, isTraining, isOptimizing):
		assert stepsPerEpoch > 0
		# No sense to call this of a GAN if not training, use each model separately.
		assert isTraining == True

		metricResults = {
			"generator" : {metric : RunningMean() for metric in self.generator.callbacks.keys()},
			"discriminator" : {metric : RunningMean() for metric in self.discriminator.callbacks.keys()}
		}
		self.generator.linePrinter = MessagePrinter(None)
		self.discriminator.linePrinter = MessagePrinter(None)

		startTime = datetime.now()
		for i, items in enumerate(generator):
			if i == stepsPerEpoch:
				break

			(gNoise, gLabels), ((trueData, dNoise), dLabels) = items
			# First, we call the generator's methods (including callbacks and all), using the provided noise.
			metrics = self.generator.run_one_epoch(makeGenerator(gNoise, gLabels), 1, True, isOptimizing)
			for k in metricResults["generator"]:
				metricResults["generator"][k].update(metrics[k], 1)

			# Then, once the generator was optimized, we call discriminator's methods, by creating the true/fake labels
			with tr.no_grad():
				fakeData = self.generator.npForward(dNoise)
			MB = dNoise.shape[0]
			trueLabels = np.ones(MB)
			fakeLabels = np.zeros(MB)
			trueGenerator = makeGenerator(trueData, trueLabels)
			trueMetrics = self.discriminator.run_one_epoch(trueGenerator, 1, True, isOptimizing)
			fakeGenerator = makeGenerator(fakeData, fakeLabels)
			fakeMetrics = self.discriminator.run_one_epoch(fakeGenerator, 1, True, isOptimizing)
			for k in metricResults["discriminator"]:
				metricResults["discriminator"][k].update(trueMetrics[k], 1)
				metricResults["discriminator"][k].update(fakeMetrics[k], 1)

			iterFinishTime = (datetime.now() - startTime)
			message = self.computeIterPrintMessage(i, stepsPerEpoch, metricResults, iterFinishTime)
			self.linePrinter.print(message)

		# Get the values at end of epoch.
		for k in ["generator", "discriminator"]:
			for metric in metricResults[k]:
				metricResults[k][metric] = metricResults[k][metric].get()
		return metricResults
