import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
from typing import List, Union, Dict

from .network_serializer import NetworkSerializer
from .utils import getNumParams, getOptimizerStr, npGetData, trGetData, StorePrevState
from ..utilities import makeGenerator, MessagePrinter, isBaseOf, RunningMean, \
	topologicalSort, deepCheckEqual, getFormattedStr
from ..callbacks import Callback, MetricAsCallback
from ..metrics import Metric

np.set_printoptions(precision=3, suppress=True)

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NeuralNetworkPyTorch(nn.Module):
	def __init__(self, hyperParameters={}):
		assert type(hyperParameters) == dict
		self.optimizer = None
		self.optimizerScheduler = None
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
		self.serializer = NetworkSerializer(self)
		# A dictionary that holds values used to instantaite this module that should not change during training. This
		#  will be used to compare loaded models which technically hold same weights, but are different in important
		#  hyperparameters/training procedure etc. A model is identical to a saved one both if weights and important
		#  hyperparameters match exactly (i.e. SfmLearner using 1 warping image vs using 2 warping images vs using
		#  explainability mask).
		self.hyperParameters = hyperParameters
		super(NeuralNetworkPyTorch, self).__init__()

	##### Metrics, callbacks and hyperparameters #####

	# @brief Adds hyperparameters to the dictionary. Only works if the model has not been trained yet (so we don't)
	#  update the hyperparameters post-training as it would invalidate the whole principle.
	def addHyperParameters(self, hyperParameters):
		assert self.currentEpoch == 1, "Can only add hyperparameters before training"
		for key in hyperParameters:
			assert not key in self.hyperParameters
			self.hyperParameters[key] = hyperParameters[key]

	# Sets the user provided list of metrics as callbacks and adds them to the callbacks list.
	def addMetrics(self, metrics : Dict[str, Union[Metric, MetricAsCallback]]):
		assert not "Loss" in metrics, "Cannot overwrite Loss metric. This is added by default for all networks."
		assert isBaseOf(metrics, dict), "Metrics must be provided as Str=>Callback dictionary"

		for key in metrics:
			# assert hasattr(metrics[key], "__call__"), "The user provided metric %s must be callable" % (key)
			assert key not in self.callbacks, "Metric %s already in callbacks list." % (key)

			metricAsCallback = metrics[key]
			# There may be cases where the metric is already a MetricAsCallback (i.e. graph metrics inherited from each
			#  edges or simply creating the MetricAsCallback object yourself), so there is no need to wrap it twice.
			if not isBaseOf(metricAsCallback, MetricAsCallback):
				metricAsCallback = MetricAsCallback(metricName=key, metric=metricAsCallback) # type: ignore
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

			if isBaseOf(callback, MetricAsCallback):
				self.iterPrintMessageKeys.append(callback.name)
		# See warning for add Metrics
		self.topologicalSort = np.arange(len(self.callbacks))
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]

	# TODO: Add clearMetrics. Store dependencies, so we can call topological sort before/after clearCallbacks.
	# Store dependenceis on model store. Make clearCallbacks clear only callbacks, not metrics as well.

	# Returns only the callbacks that are of subclass Callback (not metrics)
	def getCallbacks(self):
		callbacks = {k : v for k, v in self.callbacks.items() if not isBaseOf(v, MetricAsCallback)}
		return callbacks

	def clearCallbacks(self):
		self.callbacks = OrderedDict({"Loss" : MetricAsCallback("Loss", lambda y, t, **k : k["loss"])})
		self.topologicalSort = np.array([0], dtype=np.uint8)
		self.topologicalKeys = np.array(["Loss"], dtype=str)
		self.iterPrintMessageKeys = ["Loss"]

	def getMetrics(self):
		callbacks = {k : v for k, v in self.callbacks.items() if isBaseOf(v, MetricAsCallback)}
		return callbacks

	# Does a topological sort on the given list of callback dependencies. This MUST be called after all addMetrics and
	#  addCallbacks are called, as these functions invalidate the topological sort.
	def setCallbacksDependencies(self, dependencies):
		dependencies = deepcopy(dependencies)
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

	# Other neural network architectures can update these
	def callbacksOnEpochStart(self, isTraining):
		# Call onEpochStart here, using only basic args
		if self.trainHistory != [] and len(self.trainHistory) >= self.currentEpoch:
			trainHistory = self.trainHistory[self.currentEpoch - 1]
		else:
			trainHistory = None

		for key in self.callbacks:
			self.callbacks[key].onEpochStart(model=self, epoch=self.currentEpoch, \
				trainHistory=trainHistory, isTraining=isTraining)

	def callbacksOnEpochEnd(self, isTraining):
		# epochResults is updated at each step in the order of topological sort
		epochResults = {}
		for key in self.topologicalKeys:
			epochResults[key] = self.callbacks[key].onEpochEnd(model=self, epoch=self.currentEpoch, \
				trainHistory=self.trainHistory, epochResults=epochResults, \
				isTraining=isTraining)

	def callbacksOnIterationStart(self, isTraining, isOptimizing):
		for key in self.callbacks:
			self.callbacks[key].onIterationStart(isTraining=isTraining, isOptimizing=isOptimizing)

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		iterResults = {}
		metrics = self.getMetrics()
		for key in self.topologicalKeys:
			# iterResults is updated at each step in the order of topological sort
			result = self.callbacks[key].onIterationEnd(results, labels, data=data, loss=loss, \
				iteration=iteration, numIterations=numIterations, iterResults=iterResults, \
				metricResults=metricResults, isTraining=isTraining, isOptimizing=isOptimizing)
			iterResults[key] = self.callbacks[key].iterationReduceFunction(result)

			# Add it to running mean only if it's numeric. Here's why the metrics differ for different batch size
			#  values. There's no way for us to infer the batch of each iteration, so we assume it's 1.
			if key in metrics:
				metricResults[key].update(iterResults[key], 1)
		return metricResults

	##### Traiming / testing functions

	def mainLoop(self, npInputs, npLabels, isTraining=False, isOptimizing=False):
		trInputs, trLabels = trGetData(npInputs), trGetData(npLabels)
		self.iterationEpilogue(isTraining, isOptimizing, trLabels)

		# Call the network algorithm. By default this is just results = self.forward(inputs);
		#  loss = criterion(results). But this can be updated for specific network architectures (i.e. GANs)
		trResults, trLoss = self.networkAlgorithm(trInputs, trLabels)

		npResults, npLoss = npGetData(trResults), npGetData(trLoss)

		# Might be better to use a callback so we skip this step
		if not trLoss is None:
			if isTraining and isOptimizing:
				self.optimizer.zero_grad()
				trLoss.backward()
				self.optimizer.step()
			else:
				trLoss.detach_()

		return npResults, npLoss

	def initializeEpochMetrics(self):
		return {name : RunningMean(initValue=metric.defaultValue()) for name, metric in self.getMetrics().items()}

	def reduceEpochMetrics(self, metricResults):
		results = {}
		# Get the values at end of epoch. Also, apply the reduceFunction for complex MetricAsCallbacks.
		for metric in self.getMetrics():
			result = metricResults[metric].get()
			results[metric] = self.callbacks[metric].epochReduceFunction(result)
		return results

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, isTraining, isOptimizing):
		assert stepsPerEpoch > 0
		if isOptimizing == False and tr.is_grad_enabled():
			print("Warning! Not optimizing, but grad is enabled.")
		if isTraining and isOptimizing:
			assert not self.optimizer is None, "Set optimizer before training"
		assert not self.criterion is None, "Set criterion before training or testing"
		metricResults = self.initializeEpochMetrics()

		# The protocol requires the generator to have 2 items, inputs and labels (both can be None). If there are more
		#  inputs, they can be packed together (stacked) or put into a list, in which case the ntwork will receive the
		#  same list, but every element in the list is tranasformed in torch format.
		startTime = datetime.now()
		for i, items in enumerate(generator):
			npInputs, npLabels = items
			npResults, npLoss = self.mainLoop(npInputs, npLabels, isTraining, isOptimizing)

			self.iterationPrologue(npInputs, npLabels, npResults, npLoss, i, stepsPerEpoch, \
				metricResults, isTraining, isOptimizing, startTime)

			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			self.linePrinter(("Warning! Number of iterations (%d) does not match expected ") + \
				("iterations in reader (%d)") % (i, stepsPerEpoch - 1))

		res = self.reduceEpochMetrics(metricResults)
		return res

	def test_generator(self, generator, stepsPerEpoch, printMessage=None):
		assert stepsPerEpoch > 0
		self.linePrinter = MessagePrinter(printMessage)
		self.callbacksOnEpochStart(isTraining=False)
		with StorePrevState(self):
			self.eval()
			with tr.no_grad():
				resultMetrics = self.run_one_epoch(generator, stepsPerEpoch, isTraining=False, isOptimizing=False)
		self.callbacksOnEpochEnd(isTraining=False)
		return resultMetrics

	def test_model(self, data, labels, batchSize, printMessage=None):
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
		return self.test_generator(dataGenerator, stepsPerEpoch=numIterations, printMessage=printMessage)

	def epochEpilogue(self):
		self.callbacksOnEpochStart(isTraining=True)

	def epochPrologue(self, epochMetrics):
		self.linePrinter(epochMetrics["message"], reset=False)
		self.trainHistory.append(epochMetrics)
		self.callbacksOnEpochEnd(isTraining=True)
		if not self.optimizerScheduler is None:
			self.optimizerScheduler.step()
		self.currentEpoch += 1

	def iterationEpilogue(self, isTraining, isOptimizing, labels):
		self.callbacksOnIterationStart(isTraining=isTraining, isOptimizing=isOptimizing)

	# Called after every iteration steps. Usually just calls onIterationEnd for all callbacks and prints the iteration
	#  message. However, other networks can call this to update other states at the end of each iteration. Example of
	#  this is Graph, which resets the GTs for all edges before each iteration.
	def iterationPrologue(self, inputs, labels, results, loss, iteration, \
		stepsPerEpoch, metricResults, isTraining, isOptimizing, startTime):
		# metrics and callbacks are merged. Each callback/metric can have one or more "parents" which
		#  forms an ayclical graph. They must be called in such an order that all the parents are satisfied before
		#  all children (topological sort).
		# Iteration callbacks are called here. These include metrics or random callbacks such as plotting results
		#  in testing mode.
		self.callbacksOnIterationEnd(data=inputs, labels=labels, results=results, \
			loss=loss, iteration=iteration, numIterations=stepsPerEpoch, metricResults=metricResults, \
			isTraining=isTraining, isOptimizing=isOptimizing)

		# Print the message, after the metrics are updated.
		iterFinishTime = (datetime.now() - startTime)
		message = self.computeIterPrintMessage(iteration, stepsPerEpoch, metricResults, iterFinishTime)
		self.linePrinter(message)

	# @param[in] generator Generator which is used to get items for numEpochs epochs, each taking stepsPerEpoch steps
	# @param[in] stepsPerEpoch How many steps each epoch takes (assumed constant). The generator must generate this
	#  amount of items every epoch.
	# @param[in] numEpochs The number of epochs the network is trained for
	# @param[in] callbacks A list of callbacks (which must be of type Callback), that implement one of the
	#  oneIterationStart, onIterationEnd, onEpochStart or onEpochEnd methods. Moreover, whenever this method is called
	#  the list is stored in this object, such that the state of each callback is stored . Moreover, if None is given,
	#  then the already stored member is used (helpful for load_models, so we don't do callbacks=model.callbacks).
	def train_generator(self, generator, stepsPerEpoch, numEpochs, validationGenerator=None, \
		validationSteps=0, printMessage="v2"):
		assert stepsPerEpoch > 0
		self.linePrinter = MessagePrinter(printMessage)

		if self.currentEpoch > numEpochs:
			self.linePrinter("Warning. Current epoch (%d) <= requested epochs (%d). Doing nothing.\n" % \
				(self.currentEpoch, numEpochs))
			return
		self.linePrinter("Training for %d epochs starting from epoch %d\n" % \
			(numEpochs - self.currentEpoch + 1, self.currentEpoch - 1))

		while self.currentEpoch <= numEpochs:
			# Add this epoch to the trainHistory list, which is used to track history
			# self.trainHistory.append({})
			assert len(self.trainHistory) == self.currentEpoch - 1
			epochMetrics = {}
			self.epochEpilogue()

			# Run for training data and append the results
			now = datetime.now()
			# No iteration callbacks are used if there is a validation set (so iteration callbacks are only
			#  done on validation set). If no validation set is used, the iteration callbacks are used on train set.
			# trainCallbacks = [] if validationGenerator != None else callbacks
			with StorePrevState(self):
				# self.train()
				epochMetrics["Train"] = self.run_one_epoch(generator, stepsPerEpoch, isTraining=True, \
					isOptimizing=True)

			# Run for validation data and append the results
			epochMetrics["Validation"] = None
			if not validationGenerator is None:
				with StorePrevState(self):
					self.eval()
					with tr.no_grad():
						epochMetrics["Validation"] = self.run_one_epoch(validationGenerator, validationSteps, \
							isTraining=True, isOptimizing=False)

			epochMetrics["duration"] = datetime.now() - now
			message = self.computePrintMessage(epochMetrics["Train"], epochMetrics["Validation"], \
				numEpochs, epochMetrics["duration"])
			epochMetrics["message"] = message
			self.epochPrologue(epochMetrics)

	def train_model(self, data, labels, batchSize, numEpochs, validationData=None, \
		validationLabels=None, printMessage=None):
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

	##### Printing functions #####

	def setIterPrintMessageKeys(self, Keys):
		for key in Keys:
			assert key in self.callbacks, "%s not in callbacks: %s" % (key, list(self.callbacks.keys()))
		self.iterPrintMessageKeys = Keys

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		messages = []
		message = "Epoch: %d. Iteration: %d/%d." % (self.currentEpoch, i + 1, stepsPerEpoch)
		# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
		#  the current estimation per epoch. That value minus current time is the current estimation for
		#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
		ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
		message += " ETA: %s" % (ETA)
		messages.append(message)

		if self.optimizer:
			messages.append("  - Optimizer: %s" % (getOptimizerStr(self.optimizer)))

		message = "  - Metrics."
		metricKeys = sorted(list(set(metricResults.keys())))
		Keys = list(filter(lambda x : x in self.iterPrintMessageKeys, metricKeys))
		for key in Keys:
			formattedStr = getFormattedStr(metricResults[key].get(), precision=3)
			message += " %s: %s." % (key, formattedStr)
		if len(Keys) > 0:
			messages.append(message)
		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		messages = []
		done = self.currentEpoch / numEpochs * 100
		message = "Epoch %d/%d. Done: %2.2f%%. Took: %s." % (self.currentEpoch, numEpochs, done, duration)
		messages.append(message)

		if self.optimizer:
			messages.append("  - Optimizer: %s" % (getOptimizerStr(self.optimizer)))

		if len(trainMetrics) == 0:
			return messages

		messages.append("  - Metrics:")
		printableMetrics = filter(lambda x : x in self.iterPrintMessageKeys, sorted(trainMetrics))
		trainMessage, validationMessage = "    - [Train]", "    - [Validation]"
		for metric in printableMetrics:
			formattedStr = getFormattedStr(trainMetrics[metric], precision=3)
			trainMessage += " %s: %s." % (metric, formattedStr)
			if not validationMetrics is None:
				formattedStr = getFormattedStr(validationMetrics[metric], precision=3)
				validationMessage += " %s: %s." % (metric, formattedStr)
		messages.append(trainMessage)
		if not validationMetrics is None:
			messages.append(validationMessage)
		return messages

	def metricsSummary(self):
		metrics = self.getMetrics()
		summaryStr = ""
		for metric in metrics:
			summaryStr += "\t- %s (%s)\n" % (metric, metrics[metric].getDirection())
		return summaryStr

	def summary(self):
		summaryStr = "[Model summary]\n"
		summaryStr += self.__str__() + "\n"

		numParams, numTrainable = getNumParams(self)
		summaryStr += "Parameters count: %d. Trainable parameters: %d.\n" % (numParams, numTrainable)

		summaryStr += "Hyperparameters:\n"
		for hyperParameter in self.hyperParameters:
			summaryStr += "\t- %s: %s\n" % (hyperParameter, self.hyperParameters[hyperParameter])

		summaryStr += "Metrics:\n"
		summaryStr += self.metricsSummary()

		strCallbacks = str(list(self.getCallbacks().keys()))[1 : -1]
		summaryStr += "Callbacks: %s\n" % ("None" if len(strCallbacks) == 0 else strCallbacks)

		summaryStr += "Optimizer: %s\n" % getOptimizerStr(self.optimizer)
		summaryStr += "Optimizer Scheduler: %s\n" % ("None" if not self.optimizerScheduler \
			else str(self.optimizerScheduler))

		summaryStr += "GPU: %s" % (tr.cuda.is_available())

		return summaryStr

	def __str__(self):
		return "General neural network architecture. Update __str__ in your model for more details when using summary."

	##### Misc functions #####

	# Optimizer can be either a class or an object. If it's a class, it will be instantiated on all the trainable
	#  parameters, and using the arguments in the variable kwargs. If it's an object, we will just use that object,
	#  and assume it's correct (for example if we want only some parameters to be trained this has to be used)
	# Examples of usage: model.setOptimizer(nn.Adam, lr=0.01), model.setOptimizer(nn.Adam(model.parameters(), lr=0.01))
	def setOptimizer(self, optimizer, **kwargs):
		if isinstance(optimizer, optim.Optimizer):
			self.optimizer = optimizer
		else:
			trainableParams = list(filter(lambda p : p.requires_grad, self.parameters()))
			if len(trainableParams) == 0:
				print("[setOptimizer] Warning, number of trainable parameters is 0. Doing nothing.")
				return
			self.optimizer = optimizer(trainableParams, **kwargs)
		self.optimizer.storedArgs = kwargs

	def setOptimizerScheduler(self, scheduler, **kwargs):
		assert not self.optimizer is None, "Optimizer must be set before scheduler!"
		if isinstance(scheduler, optim.lr_scheduler._LRScheduler):
			self.optimizerScheduler = scheduler
		else:
			self.optimizerScheduler = scheduler(optimizer=self.optimizer, **kwargs)
			# Some schedulers need acces to the model's object. Others, will not have this argument.
			self.optimizerScheduler.model = self
			self.optimizerScheduler.storedArgs = kwargs

	def setCriterion(self, criterion):
		self.criterion = criterion

	# Useful to passing numpy data but still returning backpropagable results
	def npForwardTrResult(self, *args, **kwargs):
		trArgs = trGetData(args)
		trKwargs= trGetData(kwargs)
		trResult = self.forward(*trArgs, **trKwargs)
		return trResult

	# Wrapper for passing numpy arrays, converting them to torch arrays, forward network and convert back to numpy
	# @param[in] x The input, which can be a numpy array, or a list/tuple/dict of numpy arrays
	# @return y The output of the network as numpy array
	def npForward(self, *args, **kwargs):
		trResult = self.npForwardTrResult(*args, **kwargs)
		npResult = npGetData(trResult)
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

	def saveWeights(self, path):
		return self.serializer.saveModel(path, stateKeys=["weights", "model_state"])

	def loadWeights(self, path, yolo=False):
		stateKeys = ["weights"]
		if yolo == False:
			stateKeys.append("model_state")
		self.serializer.loadModel(path, stateKeys=stateKeys)

	def saveModel(self, path):
		return self.serializer.saveModel(path, stateKeys=["weights", "optimizer", \
			"history_dict", "callbacks", "model_state"])

	def loadModel(self, path):
		self.serializer.loadModel(path, stateKeys=["weights", "optimizer", "history_dict", "callbacks", "model_state"])

	def onModelSave(self):
		return self.hyperParameters

	def onModelLoad(self, state):
		# if len(self.hyperParameters.keys()) != len(state.keys()):
		# 	return False

		allKeys = set(list(self.hyperParameters.keys()) + list(state.keys()))
		for key in allKeys:
			if not key in self.hyperParameters:
				return False

			if not key in state:
				print("Warning. Model has unknown state key: %s=%s, possibly added after training. Skipping." % \
					(key, str(self.hyperParameters[key])))
				continue
			loadedState = state[key]
			modelState = self.hyperParameters[key]

			if not deepCheckEqual(loadedState, modelState):
				return False
		return True

	def setTrainableWeights(self, value):
		for param in self.parameters():
		    param.requires_grad = value
	
	def isTrainable(self):
		for param in self.parameters():
			if param.requires_grad == True:
				return True
		return False