# pytorch_wrapper Wrapper on top of the default pytorch network, which includes a method for automatic training
import torch as tr
import torch.nn as nn
import numpy as np
import joblib
import sys
from torch.autograd import Variable
from copy import copy

from metrics import Accuracy, Loss

def maybeCuda(x):
	return x.cuda() if tr.cuda.is_available() and hasattr(x, "cuda") else x

def maybeCpu(x):
	return x.cpu() if tr.cuda.is_available() and hasattr(x, "cpu") else x

# Labels can be None, in that case only data is available (testing cases without labels)
def makeGenerator(data, labels, batchSize):
	while True:
		numData = data.shape[0]
		numIterations = numData // batchSize + (numData % batchSize != 0)
		for i in range(numIterations):
			startIndex = i * batchSize
			endIndex = (i + 1) * batchSize
			if not labels is None:
				yield data[startIndex : endIndex], labels[startIndex : endIndex]
			else:
				yield data[startIndex : endIndex]

def getNumParams(params):
	numParams, numTrainable = 0, 0
	for param in params:
		npParamCount = np.prod(param.data.shape)
		numParams += npParamCount
		if param.requires_grad:
			numTrainable += npParamCount
	return numParams, numTrainable

# Get all the parameters of the optimizer state, except the 'params' key, which is stored separated
def getOptimizerHyperParams(optimizer):
	assert len(optimizer.param_groups) == 1
	paramGroups = optimizer.param_groups[0]
	optimizerState = {}
	for key in paramGroups:
		if key == "params":
			continue

		optimizerState[key] = maybeCpu(paramGroups[key])
	return optimizerState

def getOptimizerParamsState(optimizer):
	states = []
	# Just iterate through values, because keys are the weights themselves, and we already save those.
	for param_state in list(optimizer.state.values()):
		# optimizer.state :: [param -> param_state]
		# param_state :: {Str -> state_tensor}
		saved_state = {}
		for key in param_state:
			saved_state[key] = maybeCpu(param_state[key])
		states.append(saved_state)
	return states

def getOptimizerStr(optimizer):
	groups = optimizer.param_groups[0]
	if type(optimizer) == tr.optim.SGD:
		optimizerType = "SGD"
		params = "Learning rate: %s, Momentum: %s, Dampening: %s, Weight Decay: %s, Nesterov: %s" % (groups["lr"], \
			groups["momentum"], groups["dampening"], groups["weight_decay"], groups["nesterov"])
	elif type(optimizer) == tr.optim.Adam:
		optimizerType = "Adam"
		params = "Learning rate: %s, Betas: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], groups["betas"], \
			groups["eps"], groups["weight_decay"])
	else:
		raise NotImplementedError("Not yet implemneted optimizer str for %s" % (type(optimizer)))
	return "%s. %s" % (optimizerType, params)

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NeuralNetworkPyTorch(nn.Module):
	def __init__(self):
		self.optimizer = None
		self.criterion = None
		self.metrics = {"Loss" : Loss()}
		self.startEpoch = 1
		self.isCudaEnabled = False
		super(NeuralNetworkPyTorch, self).__init__()

	def setOptimizer(self, optimizerType, **kwargs):
		trainableParams = filter(lambda p : p.requires_grad, self.parameters())
		self.optimizer = optimizerType(trainableParams, **kwargs)

	def setCriterion(self, criterion):
		self.criterion = criterion

	def setStartEpoch(self, epoch):
		assert epoch >= 1 and type(epoch) == int, "Epoch is a non-zero natural number"
		self.startEpoch = epoch

	# Set the cuda flag so when we load models, we can see if we have to load the parameters as cuda or cpu tensors
	def cuda(self):
		if tr.cuda.is_available():
			self.isCudaEnabled = True
			super().cuda()
		return self

	def cpu(self):
		self.isCudaEnabled = False
		return super().cpu()

	def setMetrics(self, metrics):
		assert "Loss" in metrics, "At least one metric is required and Loss must be in them"
		if type(metrics) in (list, tuple):
			for metric in metrics:
				if metric == "Accuracy":
					self.metrics[metric] = Accuracy()
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

		numParams, numTrainable = getNumParams(self.parameters())
		summaryStr += "Parameters count: %d. Trainable parameters: %d.\n" % (numParams, numTrainable)

		strMetrics = str(list(self.metrics.keys()))[1 : -1]
		summaryStr += "Metrics: %s\n" % (strMetrics)

		summaryStr += "Optimizer: %s\n" % getOptimizerStr(self.optimizer)

		return summaryStr

	# Handles all the initilization stuff of a specific pytorch model object.
	def setup(self):
		print("[NeuralNetworkPyTorch] Setup method is empty.")

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @param[in] optimize If true, then the optimizer is also called after each iteration
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, optimize=False, printMessage=False):
		assert "Loss" in self.metrics.keys(), "At least one metric is required and Loss must be in them"
		metricResults = {metric : 0 for metric in self.metrics.keys()}

		for i, (npData, npLabels) in enumerate(generator):
			data = maybeCuda(Variable(tr.from_numpy(npData)))
			labels = maybeCuda(Variable(tr.from_numpy(npLabels)))

			results = self.forward(data)
			npResults = maybeCpu(results.data).numpy().reshape(npLabels.shape)
			
			loss = self.criterion(results, labels)
			npLoss = maybeCpu(loss.data).numpy()

			if optimize:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			for metric in self.metrics:
				metricResults[metric] += self.metrics[metric](npResults, npLabels, loss=npLoss)

			if printMessage:
				message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
				for metric in metricResults:
					message += " %s: %2.2f." % (metric, metricResults[metric] / (i + 1))
				sys.stdout.write(message + "\r")
				sys.stdout.flush()

			del data, labels
			if i == stepsPerEpoch - 1:
				break

		for metric in metricResults:
			metricResults[metric] /= stepsPerEpoch
		return npResults, metricResults

	def train_generator(self, generator, stepsPerEpoch, numEpochs, callbacks=[], validationGenerator=None, \
		validationSteps=0):
		assert self.optimizer != None and self.criterion != None, "Set optimizer and criterion before training"
		sys.stdout.write("Training for %d epochs...\n" % (numEpochs))
		for epoch in range(self.startEpoch - 1, numEpochs):
			done = (epoch + 1) / numEpochs * 100
			message = "Epoch %d/%d. Done: %2.2f%%." % (epoch + 1, numEpochs, done)

			# Run for training data and append the results
			_, trainMetrics = self.run_one_epoch(generator, stepsPerEpoch, optimize=True, printMessage=True)
			for metric in trainMetrics:
				message += " %s: %2.2f." % (metric, trainMetrics[metric])

			# Run for validation data and append the results
			if validationGenerator != None:
				_, validationMetrics = \
					self.run_one_epoch(validationGenerator, validationSteps, optimize=False, printMessage=False)
				for metric in validationMetrics:
					message += " %s: %2.2f." % ("Val " + metric, validationMetrics[metric])

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
		validationLabels=None):
		assert self.optimizer != None and self.criterion != None, "Set optimizer and criterion before training"
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)

		validationGenerator = makeGenerator(validationData, validationLabels, validationData.shape[0]) if \
			not validationLabels is None else None

		self.train_generator(dataGenerator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks, \
			validationGenerator=validationGenerator, validationSteps=1)

	# def test_generator(self, dataGenerator, numIterations):
	# 	return self.run_one_epoch(dataGenerator, numIterations)

	# # labels can be None if not available
	# def test(self, data, labels, batchSize):
	# 	testGenerator = makeGenerator(test, labels, batchSize)
	# 	numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
	# 	return self.run_one_epoch(testGenerator, numIterations, optimize=False, printMessage=False)

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
			if self.isCudaEnabled:
				item.data = maybeCuda(params[i].data)
			else:
				item.data = params[i].data

	# Saves a complete model, consisting of weights, state and optimizer params
	def save_model(self, path):
		assert self.optimizer != None, "No optimizer was set for this model. Cannot save."
		state = {
			"params" : list(map(lambda x : x.cpu(), self.parameters())),
			"optimizer_type" : type(self.optimizer),
			"optimizer_hyper_params" : getOptimizerHyperParams(self.optimizer),
			"optimizer_params_state" : getOptimizerParamsState(self.optimizer)
		}
		tr.save(state, path)

	def load_model(self, path):
		loaded_model = tr.load(path)
		self._load_weights(loaded_model["params"])

		# Create a new instance of the optimizer. Some optimizers require a lr to be set as well
		self.setOptimizer(loaded_model["optimizer_type"], lr=0.01)

		# Load optimizer hyper parameters
		assert len(self.optimizer.param_groups) == 1
		for key in loaded_model["optimizer_hyper_params"]:
			assert key != "params"
			if self.isCudaEnabled:
				self.optimizer.param_groups[0][key] = loaded_model["optimizer_hyper_params"][key]
			else:
				self.optimizer.param_groups[0][key] = maybeCuda(loaded_model["optimizer_hyper_params"][key])

		# Load params now
		self.optimizer.param_groups[0]["params"] = list(self.parameters())

		# Load parameters state from the stored list
		for i, param in enumerate(self.optimizer.param_groups[0]["params"]):
			self.optimizer.state[param] = maybeCuda(loaded_model["optimizer_params_state"][i])
			if self.isCudaEnabled:
				for key in self.optimizer.state[param]:
					self.optimizer.state[param][key] = maybeCuda(self.optimizer.state[param][key])

		# Optimizer consistency checks
		l1 = list(self.optimizer.state_dict()["state"].keys())
		l2 = self.optimizer.state_dict()["param_groups"][0]["params"]
		l3 = list(map(lambda x : id(x), self.parameters()))
		assert l1 == l2 and l1 == l3, "Something was wrong with loading optimizer"

		print("Succesfully loaded model")