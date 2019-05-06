import os
import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

from transforms import *
import neural_wrappers.callbacks
from neural_wrappers.utilities import makeGenerator, LinePrinter, isBaseOf, RunningMean, topologicalSort
from neural_wrappers.callbacks import Callback, MetricAsCallback
from metrics import Metric

from .network_serializer import NetworkSerializer
from .pytorch_utils import maybeCuda, maybeCpu, getNumParams, getOptimizerStr, getNpData, getTrData, StorePrevState

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NeuralNetworkPyTorch(nn.Module):
	def __init__(self, hyperParameters={}):
		assert type(hyperParameters) == dict
		self.optimizer = None
		self.criterion = None
		self.currentEpoch = 1

		# Variable that holds both callbacks and metrics (which are also callbacks with just onIterationEnd method
		#   implemented). By default Loss will always be a callback
		self.callbacks = OrderedDict({"Loss" : MetricAsCallback("Loss", lambda y, t, **k : k["loss"])})
		self.topologicalSort = np.array([0], dtype=np.uint8)
		self.topologicalKeys = np.array(["Loss"], dtype=str)
		# A list of all the metrics/callbacks that are used to compute the iteration print
		#  message during training/testing
		self.iterPrintMessageKeys = ["Loss"]

		# A list that stores various information about the model at each epoch. The index in the list represents the
		#  epoch value. Each value of the list is a dictionary that holds by default only loss value, but callbacks
		#  can add more items to this (like confusion matrix or accuracy, see mnist example).
		self.trainHistory = []
		self.linePrinter = LinePrinter()
		self.serializer = NetworkSerializer(self)
		# A dictionary that holds values used to instantaite this module that should not change during training. This
		#  will be used to compare loaded models which technically hold same weights, but are different in important
		#  hyperparameters/training procedure etc. A model is identical to a saved one both if weights and important
		#  hyperparameters match exactly (i.e. SfmLearner using 1 warping image vs using 2 warping images vs using
		#  explainability mask).
		self.hyperParameters = hyperParameters
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
			trainableParams = list(filter(lambda p : p.requires_grad, self.parameters()))
			self.optimizer = optimizer(trainableParams, **kwargs)

	def setCriterion(self, criterion):
		self.criterion = criterion

	# Sets the user provided list of metrics as callbacks and adds them to the callbacks list.
	def addMetrics(self, metrics):
		assert not "Loss" in metrics, "Cannot overwrite Loss metric. This is added by default for all networks."
		assert isBaseOf(metrics, dict), "Metrics must be provided as Str=>Callback dictionary"

		for key in metrics:
			assert type(key) == str, "The key of the metric must be a string"
			assert hasattr(metrics[key], "__call__"), "The user provided transformation %s must be callable" % (key)
			assert key not in self.callbacks, "Metric %s already in callbacks list." % (key)
			# If this is a Metric object, then store it as is, because in this way, we can call it by __call__, but
			#  also this means that we can store it's fields of the object which can be meaningful, such in the case
			#  of Accuracy class.
			if isBaseOf(metrics[key], Metric):
				metricAsCallback = MetricAsCallback(metricName=key, metric=metrics[key])
			# Otherwise, this is a lambda callback, so we send the call itself to the constructor. Very important that
			#  we need to rewrite the name of the arguments as results, labels, kwargs, because the constructor of
			#  MetricAsCallback needs to call them with that notation so metrics and callbacks can be unified.
			else:
				metricAsCallback = MetricAsCallback(metricName=key, \
					metric=lambda results, labels, **kwargs : metrics[key](results, labels, **kwargs))
			# Either way, store the resulting Callback object in the list of callbacks
			self.callbacks[key] = metricAsCallback
			self.iterPrintMessageKeys.append(key)

		# Warning, calling addMetrics might invalidat topological sort as we reset the indexes here. If there are
		#  dependencies already set using setCallbacksDependeices, it must be called again.
		self.topologicalSort = np.arange(len(self.callbacks))
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]

	# Adds the user provided list of callbacks to the model's list of callbacks (and metrics)
	def addCallbacks(self, callbacks):
		for callback in callbacks:
			assert isBaseOf(callback, Callback), \
				"Expected only subclass of types Callback, got type %s" % (type(callback))
			assert callback.name not in self.callbacks, "Callback %s already in callbacks list." % (callback.name)
			self.callbacks[callback.name] = callback

		# See warning for add Metrics
		self.topologicalSort = np.arange(len(self.callbacks))
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]

	# Returns only the callbacks that are of subclass Callback (not metrics)
	def getCallbacks(self):
		callbacks = {k : v for k, v in self.callbacks.items() \
			if not isBaseOf(v, neural_wrappers.callbacks.MetricAsCallback)}
		return callbacks

	def getMetrics(self):
		callbacks = {k : v for k, v in self.callbacks.items() \
			if isBaseOf(v, neural_wrappers.callbacks.MetricAsCallback)}
		return callbacks

	# Does a topological sort on the given list of callback dependencies. This MUST be called after all addMetrics and
	#  addCallbacks are called, as these functions invalidate the topological sort.
	def setCallbacksDependencies(self, dependencies):
		for key in dependencies:
			if not key in self.callbacks:
				assert False, "Key %s is not in callbacks (%s)" % (key, list(self.callbacks.keys()))
			for depKey in dependencies[key]:
				if not depKey in self.callbacks:
					assert False, "Key %s of dependency %s is not in callbacks (%s)" \
						% (depKey, key, list(self.callbacks.keys()))

		for key in self.callbacks:
			if not key in dependencies:
				dependencies[key] = []

		order = topologicalSort(dependencies)
		callbacksKeys = list(self.callbacks.keys())
		self.topologicalSort = np.array([callbacksKeys.index(x) for x in order])
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]
		print("Successfully done topological sort!")

	def getTrainableParameters(self):
		return list(filter(lambda p : p.requires_grad, self.parameters()))

	def summary(self):
		summaryStr = "[Model summary]\n"
		summaryStr += self.__str__() + "\n"

		numParams, numTrainable = getNumParams(self.parameters())
		summaryStr += "Parameters count: %d. Trainable parameters: %d.\n" % (numParams, numTrainable)

		strHyperParameters = " | ".join(["%s => %s" % (x, y) for x, y in \
			zip(self.hyperParameters.keys(), self.hyperParameters.values())])
		summaryStr += "Hyperparameters: %s\n" % (strHyperParameters)

		strMetrics = str(list(self.getMetrics().keys()))[1 : -1]
		summaryStr += "Metrics: %s\n" % (strMetrics)

		strCallbacks = str(list(self.getCallbacks().keys()))[1 : -1]
		summaryStr += "Callbacks: %s\n" % (strCallbacks)

		summaryStr += "Optimizer: %s\n" % getOptimizerStr(self.optimizer)

		return summaryStr

	def __str__(self):
		return "General neural network architecture. Update __str__ in your model for more details when using summary."

	def populateHistoryDict(self, message, **kwargs):
		assert not kwargs["trainHistory"] is None
		trainHistory = kwargs["trainHistory"]
		trainHistory["duration"] = kwargs["duration"]
		trainHistory["trainMetrics"] = deepcopy(kwargs["trainMetrics"])
		trainHistory["validationMetrics"] = deepcopy(kwargs["validationMetrics"])
		trainHistory["message"] = message

	# Other neural network architectures can update these
	def callbacksOnEpochStart(self):
		# Call onEpochStart here, using only basic args

		if self.trainHistory != [] and len(self.trainHistory) >= self.currentEpoch:
			trainHistory = self.trainHistory[self.currentEpoch - 1]
		else:
			trainHistory = None

		for key in self.callbacks:
			self.callbacks[key].onEpochStart(model=self, epoch=self.currentEpoch, trainHistory=trainHistory)

	def callbacksOnIterationStart(self, **kwargs):
		for key in self.callbacks:
			self.callbacks[key].onIterationStart(**kwargs)

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, metricResults):
		iterResults = {}
		for i, key in enumerate(self.topologicalKeys):
			# iterResults is updated at each step in the order of topological sort
			iterResults[key] = self.callbacks[key].onIterationEnd(results, labels, data=data, loss=loss, \
				iteration=iteration, numIterations=numIterations, iterResults=iterResults, metricResults=metricResults)

			# Add it to running mean only if it's numeric
			try:
				metricResults[key].update(iterResults[key], 1)
			except Exception:
				continue

	def npForward(self, x):
		trInput = getTrData(x)
		trResult = self.forward(trInput)
		npResult = getNpData(trResult)
		return npResult

	# The network algorithm. This must be updated for specific networks, so the whole metric/callbacks system works
	#  just as before.
	# @param[in] trInputs The inputs of the network
	# @param[in] trLabels The labels of the network
	# @return The results of the network and the loss as given by the criterion function
	def networkAlgorithm(self, trInputs, trLabels):
		trResults = self.forward(trInputs)
		trLoss = self.criterion(trResults, trLabels)
		return trResults, trLoss

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, printMessage=False):
		assert stepsPerEpoch > 0
		if tr.is_grad_enabled():
			assert not self.optimizer is None, "Set optimizer before training"
		assert not self.criterion is None, "Set criterion before training or testing"
		# assert "Loss" in self.metrics.keys(), "Loss metric was not found in metrics."
		# self.checkCallbacks(callbacks)
		self.callbacksOnEpochStart()

		metricResults = {metric : RunningMean() for metric in self.callbacks.keys()}
		i = 0

		if tr.is_grad_enabled():
			optimizeCallback = (lambda optim, loss : (optim.zero_grad(), loss.backward(), optim.step()))
		else:
			optimizeCallback = (lambda optim, loss : loss.detach_())

		# The protocol requires the generator to have 2 items, inputs and labels (both can be None). If there are more
		#  inputs, they can be packed together (stacked) or put into a list, in which case the ntwork will receive the
		#  same list, but every element in the list is tranasformed in torch format.
		startTime = datetime.now()
		for i, items in enumerate(generator):
			self.callbacksOnIterationStart()
			npInputs, npLabels = items
			trInputs = getTrData(npInputs)
			trLabels = getTrData(npLabels)

			# Call the network algorithm. By default this is just results = self.forward(inputs);
			#  loss = criterion(results). But this can be updated for specific network architectures (i.e. GANs)
			trResults, trLoss = self.networkAlgorithm(trInputs, trLabels)

			npResults = getNpData(trResults)
			npLoss = maybeCpu(trLoss.detach()).numpy()
			optimizeCallback(self.optimizer, trLoss)
			iterFinishTime = (datetime.now() - startTime)

			# metrics and callbacks are merged. Each callback/metric can have one or more "parents" which
			#  forms an ayclical graph. They must be called in such an order that all the parents are satisfied before
			#  all children (topological sort).
			# Iteration callbacks are called here. These include metrics or random callbacks such as plotting results
			#  in testing mode.
			self.callbacksOnIterationEnd(data=npInputs, labels=npLabels, results=npResults, loss=npLoss, iteration=i, \
				numIterations=stepsPerEpoch, metricResults=metricResults)

			# Print the message, after the metrics are updated.
			if printMessage:
				self.linePrinter.print(self.computeIterPrintMessage(i, stepsPerEpoch, metricResults, iterFinishTime))

			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			sys.stderr.write("Warning! Number of iterations (%d) does not match expected iterations in reader (%d)" % \
				(i, stepsPerEpoch - 1))

		for metric in metricResults:
			metricResults[metric] = metricResults[metric].get()
		return metricResults

	def test_generator(self, generator, stepsPerEpoch, printMessage=False):
		assert stepsPerEpoch > 0
		now = datetime.now()
		# Store previous state and restore it after running epoch in eval mode.
		with StorePrevState(self):
			# self.eval()
			with tr.no_grad():
				resultMetrics = self.run_one_epoch(generator, stepsPerEpoch, printMessage=printMessage)
		duration = datetime.now() - now

		# Do the callbacks for the end of epoch.
		callbackArgs = {
			"model" : self,
			"epoch" : 1,
			"numEpochs" : 1,
			"trainMetrics" : None,
			"validationMetrics" : resultMetrics,
			"duration" : duration,
			"trainHistory" : None # TODO, see what to do for case where I load a model with existing history
		}

		# Add basic value to the history dictionary (just loss and time)
		for callback in callbacks:
			callback.onEpochEnd(**callbackArgs)

		return resultMetrics

	def test_model(self, data, labels, batchSize, printMessage=False):
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
		return self.test_generator(dataGenerator, stepsPerEpoch=numIterations, printMessage=printMessage)

	def setIterPrintMessageKeys(self, Keys):
		for key in Keys:
			assert key in self.callbacks, "%s not in callbacks: %s" % (key, list(self.callbacks.keys()))
		self.iterPrintMessageKeys = Keys

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
		for key in sorted(metricResults):
			if not key in self.iterPrintMessageKeys:
				continue
			message += " %s: %2.2f." % (key, metricResults[key].get())

		if self.optimizer:
			message += " LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])

		# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
		#  the current estimation per epoch. That value minus current time is the current estimation for
		#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
		ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
		message += " ETA: %s" % (ETA)
		return message

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, **kwargs):
		currentEpoch = kwargs["epoch"]
		numEpochs = kwargs["numEpochs"]
		trainMetrics = kwargs["trainMetrics"]
		validationMetrics = kwargs["validationMetrics"]
		duration = kwargs["duration"]

		done = currentEpoch / numEpochs * 100
		message = "Epoch %d/%d. Done: %2.2f%%." % (currentEpoch, numEpochs, done)
		for metric in sorted(trainMetrics):
			if not metric in self.iterPrintMessageKeys:
				continue
			message += " %s: %2.2f." % (metric, trainMetrics[metric])

		if not validationMetrics is None:
			for metric in sorted(validationMetrics):
				if not metric in self.iterPrintMessageKeys:
					continue
				message += " %s: %2.2f." % ("Val " + metric, validationMetrics[metric])

		if self.optimizer:
			message += " LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])

		message += " Took: %s." % (duration)

		return message

	# @param[in] generator Generator which is used to get items for numEpochs epochs, each taking stepsPerEpoch steps
	# @param[in] stepsPerEpoch How many steps each epoch takes (assumed constant). The generator must generate this
	#  amount of items every epoch.
	# @param[in] numEpochs The number of epochs the network is trained for
	# @param[in] callbacks A list of callbacks (which must be of type Callback), that implement one of the
	#  oneIterationStart, onIterationEnd, onEpochStart or onEpochEnd methods. Moreover, whenever this method is called
	#  the list is stored in this object, such that the state of each callback is stored . Moreover, if None is given,
	#  then the already stored member is used (helpful for load_models, so we don't do callbacks=model.callbacks).
	def train_generator(self, generator, stepsPerEpoch, numEpochs, validationGenerator=None, \
		validationSteps=0, printMessage=True, **kwargs):
		assert stepsPerEpoch > 0

		# Callbacks validation and storing (for save_model)
		# if callbacks == None:
			# callbacks = self.callbacks
		# self.checkCallbacks(callbacks)
		# self.callbacks = callbacks

		if self.currentEpoch > numEpochs:
			sys.stdout.write("Warning. Current epoch (%d) <= requested epochs (%d). Doing nothing.\n" \
				% (self.currentEpoch, numEpochs))
			return

		if printMessage:
			sys.stdout.write("Training for %d epochs starting from epoch %d\n" % (numEpochs - self.currentEpoch + 1, \
				self.currentEpoch - 1))

		while self.currentEpoch <= numEpochs:
			# Add this epoch to the trainHistory list, which is used to track history
			self.trainHistory.append({})
			assert len(self.trainHistory) == self.currentEpoch

			# Run for training data and append the results
			now = datetime.now()
			# No iteration callbacks are used if there is a validation set (so iteration callbacks are only
			#  done on validation set). If no validation set is used, the iteration callbacks are used on train set.
			# trainCallbacks = [] if validationGenerator != None else callbacks
			with StorePrevState(self):
				# self.train()
				trainMetrics = self.run_one_epoch(generator, stepsPerEpoch,	printMessage=printMessage, **kwargs)

			# Run for validation data and append the results
			if validationGenerator != None:
				with StorePrevState(self):
					# self.eval()
					with tr.no_grad():
						validationMetrics = self.run_one_epoch(validationGenerator, validationSteps, \
							printMessage=False, **kwargs)
			duration = datetime.now() - now

			# Do the callbacks for the end of epoch.
			callbackArgs = {
				"model" : self,
				"epoch" : self.currentEpoch,
				"numEpochs" : numEpochs,
				"trainMetrics" : trainMetrics,
				"validationMetrics" : validationMetrics if validationGenerator != None else None,
				"duration" : duration,
				"trainHistory" : self.trainHistory[self.currentEpoch - 1]
			}

			# Print message is also computed in similar fashion using callback arguments
			message = self.computePrintMessage(**callbackArgs)
			if printMessage:
				sys.stdout.write(message + "\n")
				sys.stdout.flush()

			# # Add basic value to the history dictionary (just loss and time)
			self.populateHistoryDict(message, **callbackArgs)
			for key in self.callbacks:
				self.callbacks[key].onEpochEnd(**callbackArgs)

			self.currentEpoch += 1

	def train_model(self, data, labels, batchSize, numEpochs, validationData=None, \
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

		self.train_generator(dataGenerator, stepsPerEpoch=numIterations, numEpochs=numEpochs, \
			validationGenerator=validationGenerator, validationSteps=valNumIterations, printMessage=printMessage)

	def saveWeights(self, path):
		return self.serializer.saveModel(path, stateKeys=["weights", "model_state"])

	def loadWeights(self, path):
		self.serializer.loadModel(path, stateKeys=["weights", "model_state"])

	def saveModel(self, path):
		return self.serializer.saveModel(path, stateKeys=["weights", "optimizer", \
			"history_dict", "callbacks", "model_state"])

	def loadModel(self, path):
		self.serializer.loadModel(path, stateKeys=["weights", "optimizer", "history_dict", "callbacks", "model_state"])

	def onModelSave(self):
		return self.hyperParameters

	def onModelLoad(self, state):
		if len(self.hyperParameters.keys()) != len(state.keys()):
			return False

		for key in state:
			if not key in self.hyperParameters:
				return False

			if not state[key] == self.hyperParameters[key]:
				return False

		return True