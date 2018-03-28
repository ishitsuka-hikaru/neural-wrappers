import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from neural_wrappers.transforms import *
from neural_wrappers.metrics import Accuracy, Loss
from neural_wrappers.utilities import makeGenerator, LinePrinter
from .utils import maybeCuda, maybeCpu, getNumParams, getOptimizerStr

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NeuralNetworkPyTorch(nn.Module):
	def __init__(self):
		self.optimizer = None
		self.criterion = None
		self.metrics = {"Loss" : Loss()}
		self.startEpoch = 1
		super(NeuralNetworkPyTorch, self).__init__()

	### Various setters for the network ###
	# Optimizer can be either a class or an object. If it's a class, it will be instantiated on all the trainable
	#  parameters, and using the arguments in the variable kwargs. If it's an object, we will just use that object,
	#  and assume it's correct (for example if we want only some parameters to be trained this has to be used)
	# Examples of usage: model.setOptimizer(nn.Adam, lr=0.01), model.setOptimizer(nn.Adam(model.parameters(), lr=0.01))
	def setOptimizer(self, optimizer, **kwargs):
		if isinstance(optimizer, optim.Optimizer):
			self.optimizer = optimizer
		else:
			trainableParams = filter(lambda p : p.requires_grad, self.parameters())
			self.optimizer = optimizer(trainableParams, **kwargs)

	def setCriterion(self, criterion):
		self.criterion = criterion

	def setStartEpoch(self, epoch):
		assert epoch >= 1 and type(epoch) == int, "Epoch is a non-zero natural number"
		self.startEpoch = epoch

	def setMetrics(self, metrics):
		assert "Loss" in metrics, "At least one metric is required and Loss must be in them"
		if type(metrics) in (list, tuple):
			for metric in metrics:
				if metric == "Accuracy":
					self.metrics[metric] = Accuracy(categoricalLabels=True)
				elif metric == "Loss":
					self.metrics[metric] = Loss()
				else:
					raise NotImplementedError("Unknown metric provided: " + metric + ". Use dict and implementation")
		else:
			for key in metrics:
				if not type(key) is str:
					raise Exception("The key of the metric must be a string")
			self.metrics = metrics

	def summary(self):
		summaryStr = "[Model summary]\n"
		summaryStr += self.__str__() + "\n"

		numParams, numTrainable = getNumParams(self.parameters())
		summaryStr += "Parameters count: %d. Trainable parameters: %d.\n" % (numParams, numTrainable)

		strMetrics = str(list(self.metrics.keys()))[1 : -1]
		summaryStr += "Metrics: %s\n" % (strMetrics)

		summaryStr += "Optimizer: %s\n" % getOptimizerStr(self.optimizer)

		return summaryStr

	def __str__(self):
		return "General neural network architecture. Update __str__ in your model for more details when using summary."

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @param[in] optimize If true, then the optimizer is also called after each iteration
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, callbacks=[], optimize=False, printMessage=False, debug=False):
		assert "Loss" in self.metrics.keys(), "At least one metric is required and Loss must be in them"
		assert not self.criterion is None, "Expected a criterion/loss to be set before training/testing."
		metricResults = {metric : 0 for metric in self.metrics.keys()}
		linePrinter = LinePrinter()
		i = 0
		for i, (npData, npLabels) in enumerate(generator):
			data = maybeCuda(Variable(tr.from_numpy(npData)))
			labels = maybeCuda(Variable(tr.from_numpy(npLabels)))

			results = self.forward(data)
			npResults = maybeCpu(results.data).numpy()
			
			loss = self.criterion(results, labels)
			npLoss = maybeCpu(loss.data).numpy()
			if debug:
				print("\nLoss: %2.6f" % (npLoss))

			if optimize:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			# Iteration callbacks are called here (i.e. for plotting results!)
			callbackArgs = {
				"data" : npData,
				"labels" : npLabels,
				"results" : npResults,
				"loss" : npLoss,
				"iteration" : i,
				"numIterations" : stepsPerEpoch
			}
			for callback in callbacks:
				callback(**callbackArgs)

			for metric in self.metrics:
				metricResults[metric] += self.metrics[metric](npResults, npLabels, loss=npLoss)

			if printMessage:
				message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
				for metric in metricResults:
					message += " %s: %2.2f." % (metric, metricResults[metric] / (i + 1))
				linePrinter.print(message)

			del data, labels, results, npData, npLabels, npResults
			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			sys.stderr.write("Warning! Number of iterations (%d) does not match expected iterations in reader (%d)" % \
				(i, stepsPerEpoch - 1))
		for metric in metricResults:
			metricResults[metric] /= stepsPerEpoch
		return metricResults

	def test_generator(self, generator, stepsPerEpoch, callbacks=[], printMessage=False):
		return self.run_one_epoch(generator, stepsPerEpoch, callbacks=callbacks, optimize=False, \
			printMessage=printMessage)

	def test_model(self, data, labels, batchSize, callbacks=[], printMessage=False):
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
		return self.test_generator(dataGenerator, stepsPerEpoch=numIterations, callbacks=callbacks, \
			printMessage=printMessage)

	def train_generator(self, generator, stepsPerEpoch, numEpochs, callbacks=[], validationGenerator=None, \
		validationSteps=0, printMessage=True):
		assert self.optimizer != None and self.criterion != None, "Set optimizer and criterion before training"
		if printMessage:
			sys.stdout.write("Training for %d epochs...\n" % (numEpochs))
	
		for epoch in range(self.startEpoch - 1, numEpochs):
			done = (epoch + 1) / numEpochs * 100
			message = "Epoch %d/%d. Done: %2.2f%%." % (epoch + 1, numEpochs, done)

			# Run for training data and append the results
			trainMetrics = self.run_one_epoch(generator, stepsPerEpoch, optimize=True, printMessage=printMessage)
			for metric in trainMetrics:
				message += " %s: %2.2f." % (metric, trainMetrics[metric])

			# Run for validation data and append the results
			if validationGenerator != None:
				validationMetrics = self.run_one_epoch(validationGenerator, validationSteps, optimize=False)
				for metric in validationMetrics:
					message += " %s: %2.2f." % ("Val " + metric, validationMetrics[metric])

			if printMessage:
				sys.stdout.write(message + "\n")
				sys.stdout.flush()

			# Do the callbacks
			callbackArgs = {
				"model" : self,
				"epoch" : epoch + 1,
				"numEpochs" : numEpochs,
				"trainMetrics" : trainMetrics,
				"validationMetrics" : validationMetrics if validationGenerator != None else None
			}
			for callback in callbacks:
				callback(**callbackArgs)

	def train_model(self, data, labels, batchSize, numEpochs, callbacks=[], validationData=None, \
		validationLabels=None, printMessage=True):
		assert self.optimizer != None and self.criterion != None, "Set optimizer and criterion before training"
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)

		if not validationData is None:
			valNumIterations = validationData.shape[0] // batchSize + (validationData.shape[0] % batchSize != 0)
			validationGenerator = makeGenerator(validationData, validationLabels, batchSize)
		else:
			valNumIterations = 1
			validationGenerator = None

		self.train_generator(dataGenerator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks, \
			validationGenerator=validationGenerator, validationSteps=valNumIterations, printMessage=printMessage)

	def save_weights(self, path):
		modelParams = list(map(lambda x : x.cpu(), self.parameters()))
		tr.save(modelParams, path)

	def load_weights(self, path):
		params = tr.load(path)
		# The file can be a weights file (just weights then) or a state file (so file["params"] are the weights)
		if type(params) == dict:
			self._load_weights(params["params"])
		else:
			self._load_weights(params)
		print("Succesfully loaded weights")

	# actual function that loads the params and checks consistency
	def _load_weights(self, params):
		loadedParams, _ = getNumParams(params)
		thisParams, _ = getNumParams(self.parameters())
		if loadedParams != thisParams:
			raise Exception("Inconsistent parameters: %d vs %d." % (loadedParams, thisParams))

		for i, item in enumerate(self.parameters()):
			if item.data.shape != params[i].data.shape:
				raise Exception("Inconsistent parameters: %d vs %d." % (item.data.shape, params[i].data.shape))
			item.data = maybeCuda(params[i].data)

	# Saves a complete model, consisting of weights, state and optimizer params
	def save_model(self, path):
		assert self.optimizer != None, "No optimizer was set for this model. Cannot save."
		state = {
			"params" : list(map(lambda x : x.cpu(), self.parameters())),
			"optimizer_type" : type(self.optimizer),
			"optimizer_state" : self.optimizer.state_dict()
		}
		tr.save(state, path)

	def load_model(self, path):
		loaded_model = tr.load(path)
		self._load_weights(loaded_model["params"])

		# Create a new instance of the optimizer. Some optimizers require a lr to be set as well
		self.setOptimizer(loaded_model["optimizer_type"], lr=0.01)
		self.optimizer.load_state_dict(loaded_model["optimizer_state"])

		# Optimizer consistency checks
		# l1 = list(self.optimizer.state_dict()["state"].keys()) # Not sure if/how we can use this (not always ordered)
		l2 = self.optimizer.state_dict()["param_groups"][0]["params"]
		l3 = list(map(lambda x : id(x), self.parameters()))
		assert l2 == l3, "Something was wrong with loading optimizer"

		print("Succesfully loaded model")