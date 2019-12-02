import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

from ..utilities import makeGenerator, MultiLinePrinter, isBaseOf, RunningMean, topologicalSort, deepCheckEqual
from ..callbacks import Callback, MetricAsCallback

from .network_serializer import NetworkSerializer
from .pytorch_utils import getNumParams, getOptimizerStr, getNpData, getTrData, StorePrevState

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
		self.linePrinter = MultiLinePrinter()
		self.serializer = NetworkSerializer(self)
		# A dictionary that holds values used to instantaite this module that should not change during training. This
		#  will be used to compare loaded models which technically hold same weights, but are different in important
		#  hyperparameters/training procedure etc. A model is identical to a saved one both if weights and important
		#  hyperparameters match exactly (i.e. SfmLearner using 1 warping image vs using 2 warping images vs using
		#  explainability mask).
		self.hyperParameters = hyperParameters
		super(NeuralNetworkPyTorch, self).__init__()

	##### Metrics and callbacks #####

	# Sets the user provided list of metrics as callbacks and adds them to the callbacks list.
	def addMetrics(self, metrics):
		assert not "Loss" in metrics, "Cannot overwrite Loss metric. This is added by default for all networks."
		assert isBaseOf(metrics, dict), "Metrics must be provided as Str=>Callback dictionary"

		for key in metrics:
			# assert type(key) == str, "The key of the metric must be a string"
			assert hasattr(metrics[key], "__call__"), "The user provided transformation %s must be callable" % (key)
			assert key not in self.callbacks, "Metric %s already in callbacks list." % (key)

			metricAsCallback = MetricAsCallback(metricName=key, metric=metrics[key])
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
		for i, key in enumerate(self.topologicalKeys):
			# iterResults is updated at each step in the order of topological sort
			iterResults[key] = self.callbacks[key].onIterationEnd(results, labels, data=data, loss=loss, \
				iteration=iteration, numIterations=numIterations, iterResults=iterResults, \
				metricResults=metricResults, isTraining=isTraining, isOptimizing=isOptimizing)

			# Add it to running mean only if it's numeric
			try:
				metricResults[key].update(iterResults[key], 1)
			except Exception:
				continue

	##### Traiming / testing functions

	def mainLoop(self, npInputs, npLabels, isTraining=False, isOptimizing=False):
		trInputs, trLabels = getTrData(npInputs), getTrData(npLabels)
		self.iterationEpilogue(isTraining, isOptimizing, trLabels)

		# Call the network algorithm. By default this is just results = self.forward(inputs);
		#  loss = criterion(results). But this can be updated for specific network architectures (i.e. GANs)
		trResults, trLoss = self.networkAlgorithm(trInputs, trLabels)

		npResults, npLoss = getNpData(trResults), getNpData(trLoss)

		# Might be better to use a callback so we skip this step
		if isTraining and isOptimizing:
			self.optimizer.zero_grad()
			trLoss.backward()
			self.optimizer.step()
		else:
			trLoss.detach_()

		return npResults, npLoss

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, isTraining, isOptimizing, printMessage=False):
		assert stepsPerEpoch > 0
		if isOptimizing == False and tr.is_grad_enabled():
			print("Warning! Not optimizing, but grad is enabled.")
		if isTraining and isOptimizing:
			assert not self.optimizer is None, "Set optimizer before training"
		assert not self.criterion is None, "Set criterion before training or testing"

		metricResults = {metric : RunningMean() for metric in self.callbacks.keys()}

		# The protocol requires the generator to have 2 items, inputs and labels (both can be None). If there are more
		#  inputs, they can be packed together (stacked) or put into a list, in which case the ntwork will receive the
		#  same list, but every element in the list is tranasformed in torch format.
		startTime = datetime.now()
		for i, items in enumerate(generator):
			npInputs, npLabels = items
			npResults, npLoss = self.mainLoop(npInputs, npLabels, isTraining, isOptimizing)

			self.iterationPrologue(npInputs, npLabels, npResults, npLoss, i, stepsPerEpoch, \
				metricResults, isTraining, isOptimizing, printMessage, startTime)

			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			sys.stderr.write("Warning! Number of iterations (%d) does not match expected iterations in reader (%d)" % \
				(i, stepsPerEpoch - 1))

		# Get the values at end of epoch.
		for metric in metricResults:
			metricResults[metric] = metricResults[metric].get()
		return metricResults

	def test_generator(self, generator, stepsPerEpoch, printMessage=False):
		assert stepsPerEpoch > 0
		self.callbacksOnEpochStart(isTraining=False)
		with StorePrevState(self):
			# self.eval()
			with tr.no_grad():
				now = datetime.now()
				# Store previous state and restore it after running epoch in eval mode.
				resultMetrics = self.run_one_epoch(generator, stepsPerEpoch, isTraining=False, \
					isOptimizing=False, printMessage=printMessage)
				duration = datetime.now() - now
		self.callbacksOnEpochEnd(isTraining=False)
		return resultMetrics

	def test_model(self, data, labels, batchSize, printMessage=False):
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
		return self.test_generator(dataGenerator, stepsPerEpoch=numIterations, printMessage=printMessage)

	def epochEpilogue(self):
		self.callbacksOnEpochStart(isTraining=True)

	def epochPrologue(self, epochMetrics, printMessage):
		epochMetrics["message"] = "\n".join(epochMetrics["message"])
		if printMessage:
			sys.stdout.write(epochMetrics["message"] + "\n")
			sys.stdout.flush()

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
		stepsPerEpoch, metricResults, isTraining, isOptimizing, printMessage, startTime):
		# metrics and callbacks are merged. Each callback/metric can have one or more "parents" which
		#  forms an ayclical graph. They must be called in such an order that all the parents are satisfied before
		#  all children (topological sort).
		# Iteration callbacks are called here. These include metrics or random callbacks such as plotting results
		#  in testing mode.
		self.callbacksOnIterationEnd(data=inputs, labels=labels, results=results, \
			loss=loss, iteration=iteration, numIterations=stepsPerEpoch, metricResults=metricResults, \
			isTraining=isTraining, isOptimizing=isOptimizing)

		# Print the message, after the metrics are updated.
		if printMessage:
			iterFinishTime = (datetime.now() - startTime)
			message = self.computeIterPrintMessage(iteration, stepsPerEpoch, metricResults, iterFinishTime)
			self.linePrinter.print(message)

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

		if self.currentEpoch > numEpochs:
			sys.stdout.write("Warning. Current epoch (%d) <= requested epochs (%d). Doing nothing.\n" \
				% (self.currentEpoch, numEpochs))
			return

		if printMessage:
			sys.stdout.write("Training for %d epochs starting from epoch %d\n" % (numEpochs - self.currentEpoch + 1, \
				self.currentEpoch - 1))

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
					isOptimizing=True, printMessage=printMessage)

			# Run for validation data and append the results
			if not validationGenerator is None:
				with StorePrevState(self):
					# self.eval()
					with tr.no_grad():
						validationMetrics = self.run_one_epoch(validationGenerator, validationSteps, \
							isTraining=True, isOptimizing=False, printMessage=False)
				epochMetrics["Validation"] = validationMetrics
			else:
				validationMetrics = None

			duration = datetime.now() - now

			epochMetrics["duration"] = duration
			message = self.computePrintMessage(epochMetrics["Train"], validationMetrics, numEpochs, duration)
			epochMetrics["message"] = message
			self.epochPrologue(epochMetrics, printMessage)

	def train_model(self, data, labels, batchSize, numEpochs, validationData=None, \
		validationLabels=None, printMessage=True):
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
		message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
		if self.optimizer:
			message += " LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])
		# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
		#  the current estimation per epoch. That value minus current time is the current estimation for
		#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
		ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
		message += " ETA: %s" % (ETA)
		messages.append(message)

		message = "  - Metrics."
		for key in sorted(metricResults):
			if not key in self.iterPrintMessageKeys:
				continue
			message += " %s: %2.3f." % (key, metricResults[key].get())
		messages.append(message)
		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		messages = []
		done = self.currentEpoch / numEpochs * 100
		message = "Epoch %d/%d. Done: %2.2f%%." % (self.currentEpoch, numEpochs, done)
		if self.optimizer:
			message += " LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])
		message += " Took: %s." % (duration)
		messages.append(message)

		message = "  - Metrics. [Train]"
		for metric in sorted(trainMetrics):
			if not metric in self.iterPrintMessageKeys:
				continue
			message += " %s: %2.3f." % (metric, trainMetrics[metric])
		if not validationMetrics is None:
			message += " | [Validation] "
			for metric in sorted(validationMetrics):
				if not metric in self.iterPrintMessageKeys:
					continue
				message += " Val %s: %2.3f." % (metric, validationMetrics[metric])
		messages.append(message)
		return messages

	def summary(self):
		summaryStr = "[Model summary]\n"
		summaryStr += self.__str__() + "\n"

		numParams, numTrainable = getNumParams(self.parameters())
		summaryStr += "Parameters count: %d. Trainable parameters: %d.\n" % (numParams, numTrainable)

		strHyperParameters = " | ".join(["%s => %s" % (x, y) for x, y in \
			zip(self.hyperParameters.keys(), self.hyperParameters.values())])
		summaryStr += "Hyperparameters: %s\n" % (strHyperParameters)

		strMetrics = str(list(self.getMetrics().keys()))[1 : -1]
		summaryStr += "Metrics: %s\n" % ("None" if len(strMetrics) == 0 else strMetrics)

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
			self.optimizer = optimizer(trainableParams, **kwargs)
		self.optimizer.storedArgs = kwargs

	def setOptimizerScheduler(self, scheduler, **kwargs):
		assert not self.optimizer is None, "Optimizer must be set before scheduler!"
		self.optimizerScheduler = scheduler(optimizer=self.optimizer, **kwargs)
		# Some schedulers need acces to the model's object. Others, will not have this argument.
		self.optimizerScheduler.model = self
		self.optimizerScheduler.storedArgs = kwargs

	def setCriterion(self, criterion):
		self.criterion = criterion

	# Useful to passing numpy data but still returning backpropagable results
	def npForwardTrResult(self, x):
		trInput = getTrData(x)
		trResult = self.forward(trInput)
		return trResult

	# Wrapper for passing numpy arrays, converting them to torch arrays, forward network and convert back to numpy
	# @param[in] x The input, which can be a numpy array, or a list/tuple/dict of numpy arrays
	# @return y The output of the network as numpy array
	def npForward(self, x):
		npResult = getNpData(self.npForwardTrResult(x))
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