import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
from typing import List, Union, Dict, Callable, Optional, Generator
from types import LambdaType
from tqdm.auto import trange, tqdm

from abc import abstractmethod, ABC
from .network_serializer import NetworkSerializer
from .utils import getNumParams, npGetData, trGetData, _getOptimizerStr
from ..utilities import makeGenerator, isBaseOf, RunningMean, topologicalSort, deepCheckEqual, \
	getFormattedStr, Debug
from ..callbacks import Callback, CallbackName
from ..metrics import Metric, MetricWrapper

np.set_printoptions(precision=3, suppress=True)

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NWModule(nn.Module, ABC):
	def __init__(self, hyperParameters={}):
		assert type(hyperParameters) == dict
		self.optimizer:Optional[optim.Optim] = None
		self.optimizerScheduler = None
		self.criterion = None
		self.currentEpoch = 1
		self.numEpochs = 0

		# Setup print message keys, callbacks & topological sort variables
		self.clearCallbacks()

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
		super(NWModule, self).__init__()

	# The network algorithm. This must be updated for specific networks, so the whole metric/callbacks system works
	#  just as before.
	# @param[in] trInputs The inputs of the network
	# @param[in] trLabels The labels of the network
	# @return The results of the network and the loss as given by the criterion function
	@abstractmethod
	def networkAlgorithm(self, trInputs, trLabels, isTraining=False, isOptimizing=False):
		pass

	##### Metrics, callbacks and hyperparameters #####

	# @brief Adds hyperparameters to the dictionary. Only works if the model has not been trained yet (so we don't)
	#  update the hyperparameters post-training as it would invalidate the whole principle.
	def addHyperParameters(self, hyperParameters):
		assert self.currentEpoch == 1, "Can only add hyperparameters before training"
		for key in hyperParameters:
			assert not key in self.hyperParameters
			self.hyperParameters[key] = hyperParameters[key]

	def invalidateTopologicalSort(self):
		# See warning for add Metrics
		self.topologicalSort = np.arange(len(self.callbacks))
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]
		self.topologicalSortDirty = True

	def addMetric(self, metricName:Union[str, CallbackName], metric:Union[Callable, Metric]):
		# partial or lambda but not metric
		if isinstance(metric, (Callable, LambdaType, Callback)) and (not isinstance(metric, Metric)): #type: ignore
			metric = MetricWrapper(metric) #type: ignore
		if not isinstance(metricName, CallbackName): #type: ignore
			metricName = CallbackName(metricName) #type: ignore
		if metricName in self.callbacks:
			raise Exception("Metric %s already exists" % (metricName))

		metric.setName(metricName) #type: ignore
		self.callbacks[metricName] = metric

	# Sets the user provided list of metrics as callbacks and adds them to the callbacks list.
	def addMetrics(self, metrics:Dict[str, Union[Callable, Metric]]):
		assert isBaseOf(metrics, dict), "Metrics must be provided as Str=>Callback dictionary"

		for metricName, metric in metrics.items():
			assert metricName not in self.callbacks, "Metric %s already in callbacks list." % (metricName)
			self.addMetric(metricName, metric)
		# Warning, calling addMetrics will invalidate topological sort as we reset the indexes here. If there are
		#  dependencies already set using setCallbacksDependeices, it must be called again.
		self.invalidateTopologicalSort()

	def addCallback(self, callback:Callback):
		assert isBaseOf(callback, Callback), \
			"Expected only subclass of types Callback, got type %s" % (type(callback))
		assert callback.name not in self.callbacks, "Callback %s already in callbacks list." % (callback.name)
		self.callbacks[callback.name] = callback
		# Warning, calling addCallbacks will invalidate topological sort as we reset the indexes here. If there are
		#  dependencies already set using setCallbacksDependeices, it must be called again.
		# TODO: see if this is needed or we can add a new node to the existing topological sort graph
		self.invalidateTopologicalSort()

	# Adds the user provided list of callbacks to the model's list of callbacks (and metrics)
	def addCallbacks(self, callbacks:List[Callback]):
		for callback in callbacks:
			self.addCallback(callback)

	# TODO: Add clearMetrics. Store dependencies, so we can call topological sort before/after clearCallbacks.
	# Store dependencies on model store. Make clearCallbacks clear only callbacks, not metrics as well.
	def clearMetrics(self):
		metrics = [x.name for x in filter(lambda x : isBaseOf(x, Metric), self.callbacks.values())]
		newCallbacks = OrderedDict({})
		for key in self.callbacks:
			callback = self.callbacks[key]
			if not isBaseOf(callback, Metric):
				newCallbacks[key] = callback
		self.callbacks = newCallbacks
		self.invalidateTopologicalSort()

	def clearCallbacks(self):
		self.callbacks = OrderedDict()
		self.addMetric("Loss", lambda y, t, **k : k["loss"])
		self.invalidateTopologicalSort()

	# Does a topological sort on the given list of callback dependencies. This MUST be called after all addMetrics and
	#  addCallbacks are called, as these functions invalidate the topological sort.
	def setCallbacksDependencies(self, dependencies):
		# Convert strings and callbacks to CallbackNames
		convertedDependencies = {}
		for key in dependencies:
			items = dependencies[key]
			if isinstance(key, str):
				key = CallbackName(key)
			if isBaseOf(key, Callback):
				key = key.getName()
			convertedDependencies[key] = []
			for item in items:
				if isinstance(item, str):
					item = CallbackName(item)
				if isBaseOf(item, Callback):
					item = item.getName()
				convertedDependencies[key].append(item)

		for key in convertedDependencies:
			items = convertedDependencies[key]

			if not key in self.callbacks:
				assert False, "Key %s is not in callbacks (%s)" % (key, list(self.callbacks.keys()))

			for depKey in items:
				if not depKey in self.callbacks:
					assert False, "Key %s of dependency %s is not in callbacks (%s)" \
						% (depKey, key, list(self.callbacks.keys()))

		for key in self.callbacks:
			assert isinstance(key, CallbackName)
			if not key in convertedDependencies:
				convertedDependencies[key] = []

		order = topologicalSort(convertedDependencies)
		callbacksKeys = list(self.callbacks.keys())
		self.topologicalSort = np.array([callbacksKeys.index(x) for x in order])
		self.topologicalKeys = np.array(list(self.callbacks.keys()))[self.topologicalSort]
		Debug.log(2, "Successfully done topological sort!")

	### Callbacks for training purposes

	# Other neural network architectures can update these
	def callbacksOnEpochStart(self, isTraining):
		# Call onEpochStart here, using only basic args
		for key in self.callbacks:
			self.callbacks[key].onEpochStart(model=self, epoch=self.currentEpoch, \
				trainHistory=self.getTrainHistory(), isTraining=isTraining)

	def callbacksOnEpochEnd(self, isTraining):
		# epochResults is updated at each step in the order of topological sort
		epochResults = {}
		for key in self.topologicalKeys:
			epochResults[key] = self.callbacks[key].onEpochEnd(model=self, epoch=self.currentEpoch, \
				trainHistory=self.getTrainHistory(), epochResults=epochResults, \
				isTraining=isTraining)

	def callbacksOnIterationStart(self, isTraining, isOptimizing):
		for key in self.callbacks:
			self.callbacks[key].onIterationStart(isTraining=isTraining, isOptimizing=isOptimizing)

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		iterResults = {}
		modelMetrics = self.getMetrics()
		# iterResults is updated at each step in the order of topological sort
		for topologicalKey in self.topologicalKeys:
			callback = self.callbacks[topologicalKey]
			callbackResult = callback.onIterationEnd(results, labels, data=data, loss=loss, \
				iteration=iteration, numIterations=numIterations, iterResults=iterResults, \
				metricResults=metricResults, isTraining=isTraining, isOptimizing=isOptimizing)
			callbackResult = callback.iterationReduceFunction(callbackResult)
			iterResults[callback] = callbackResult

			# Add it to running mean only if it's numeric. Here's why the metrics differ for different batch size
			#  values. There's no way for us to infer the batch of each iteration, so we assume it's 1.
			if callback in modelMetrics:
				assert callback.getName() in metricResults, "Metric %s not in metric results" % (metric.getName())
				metricResults[callback.getName()].update(callbackResult, count=1)
		return metricResults

	##### Traiming / testing functions

	def updateOptimizer(self, trLoss, isTraining:bool, isOptimizing:bool, retain_graph=False):
		if not trLoss is None:
			if isTraining and isOptimizing:
				assert not self.getOptimizer() is None, "Set optimizer when training"
				self.getOptimizer().zero_grad()
				trLoss.backward(retain_graph=retain_graph)
				self.getOptimizer().step()
			else:
				trLoss.detach_()

	def initializeEpochMetrics(self):
		metrics = self.getMetrics()
		names = map(lambda x : x.getName(), metrics)
		return {name : RunningMean(initValue=metric.defaultValue()) for name, metric in zip(names, metrics)}

	def reduceEpochMetrics(self, metricResults):
		results = {}
		# Get the values at end of epoch. Also, apply the reduceFunction for complex Metrics.
		for metric in self.getMetrics():
			result = metricResults[metric.name].get()
			results[metric.name] = metric.epochReduceFunction(result)
		return results

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, isTraining, isOptimizing, Prefix):
		assert stepsPerEpoch > 0
		if isOptimizing == False and tr.is_grad_enabled():
			Debug.log(2, "Warning! Not optimizing, but grad is enabled.")
		metricResults = self.initializeEpochMetrics()

		# The protocol requires the generator to have 2 items, inputs and labels (both can be None). If there are more
		#  inputs, they can be packed together (stacked) or put into a list, in which case the ntwork will receive the
		#  same list, but every element in the list is tranasformed in torch format.
		startTime = datetime.now()

		desc = "[%s] Iteration" % Prefix
		postfix = {str(k) : 0.000 for k in metricResults}
		# stepsPerEpoch = 5
		Range = Debug.range(stepsPerEpoch, desc=desc, postfix=postfix)
		for i in Range:
			# TODO: we still assume inputs, labels = data. Update this to be generic!
			items, _ = next(generator)
			npInputs, npLabels = items
			trInputs, trLabels = trGetData(npInputs), trGetData(npLabels)

			self.iterationEpilogue(isTraining, isOptimizing, trLabels)
			trResults, trLoss = self.networkAlgorithm(trInputs, trLabels, isTraining, isOptimizing)
			npResults, npLoss = npGetData(trResults), npGetData(trLoss)
			self.iterationPrologue(npInputs, npLabels, npResults, npLoss, i, stepsPerEpoch, \
				metricResults, isTraining, isOptimizing, startTime)

			# For performance reasons we update the shown loss less often.
			if Debug.use("tqdm") and (i % 30 == 0 or i == stepsPerEpoch - 1):
				# TODO: some architectures (i.e. graph) have deep results (so dict[dict[running_mean]])
				currentResults = {}
				for k in metricResults:
					if not isinstance(metricResults[k], RunningMean):
						continue
					currentResults[str(k)] = "%2.3f" % metricResults[k].get()
				Range.set_postfix(currentResults)

		res = self.reduceEpochMetrics(metricResults)
		res["duration"] = datetime.now() - startTime
		if Debug.use("tqdm"):
			Range.close()
		return res

	# @brief Tests the model given a generator
	# @param[in] The generator used to yield testing data
	def testGenerator(self, generator:Generator):
		self.epochEpilogue(isTraining=False)
		with tr.no_grad():
			N = len(generator)
			res = self.run_one_epoch(generator, N, isTraining=False, isOptimizing=False, Prefix="Test")
			epochResults = {"Test" : res}
		self.epochPrologue(epochResults, numEpochs=1, isTraining=False)
		return epochResults

	def epochEpilogue(self, isTraining):
		self.callbacksOnEpochStart(isTraining=isTraining)

	def epochPrologue(self, epochResults, numEpochs, isTraining):
		if isTraining:
			self.getTrainHistory().append(epochResults)
			if not self.optimizerScheduler is None:
				self.optimizerScheduler.step()
		self.callbacksOnEpochEnd(isTraining=isTraining)
		if isTraining:
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

	# @param[in] generator Generator which is used to get items for numEpochs epochs, each taking len(generator) steps
	# @param[in] numEpochs The number of epochs the network is trained for
	# @param[in] validationGenerator Validation Generator used to validate the results. If not provided, generator
	#  parameter is used as validation as well for various callbacks (i.e. SaveModels)
	def trainGenerator(self, generator:Generator, numEpochs:int, validationGenerator:Optional[Generator]=None):
		if self.currentEpoch > numEpochs:
			Debug.log(1, "Warning. Current epoch (%d) <= requested epochs (%d). Doing nothing.\n" % \
				(self.currentEpoch, numEpochs))
			return

		N = numEpochs - self.currentEpoch + 1
		self.numEpochs = numEpochs
		Debug.log(2, "Training for %d epochs starting from epoch %d\n" % (N, self.currentEpoch))

		Range = Debug.range(N, initial=self.currentEpoch, total=numEpochs, position=0, desc="Epoch")
		for i in Range:
			if len(self.trainHistory) != self.currentEpoch - 1:
				breakpoint()
				assert False

			self.epochEpilogue(isTraining=True)
			epochResults = {}
			# Run for training data and append the results
			# No iteration callbacks are used if there is a validation set (so iteration callbacks are only
			#  done on validation set). If no validation set is used, the iteration callbacks are used on train set.
			res = self.run_one_epoch(generator, len(generator), isTraining=True, isOptimizing=True, Prefix="Train")
			epochResults["Train"] = res

			# Run for validation data and append the results
			if not validationGenerator is None:
				prevState = self.train if self.training else self.eval
				self.eval()
				with tr.no_grad():
					res = self.run_one_epoch(validationGenerator, len(validationGenerator), \
							isTraining=True, isOptimizing=False, Prefix="Validation")
					epochResults["Validation"] = res
				prevState()

			self.epochPrologue(epochResults, numEpochs, isTraining=True)

	def metricsSummary(self):
		metrics = self.getMetrics()
		summaryStr = ""
		for metric in metrics:
			summaryStr += "\t- %s (%s)\n" % (metric.getName(), metric.getDirection())
		return summaryStr

	def callbacksSummary(self):
		callbackNames = " | ".join(list(map(lambda x : str(x.getName()), self.getCallbacks())))
		return callbackNames

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

		summaryStr += "Callbacks:\n"
		summaryStr += "\t%s\n" % (self.callbacksSummary())

		summaryStr += "Optimizer: %s\n" % self.getOptimizerStr()
		summaryStr += "Optimizer Scheduler: %s\n" % ("None" if not self.optimizerScheduler \
			else str(self.optimizerScheduler))

		summaryStr += "GPU: %s" % (tr.cuda.is_available())

		return summaryStr

	def __str__(self):
		return "General neural network architecture. Update __str__ in your model for more details when using summary."

	##### Misc functions #####

	# Useful to passing numpy data but still returning backpropagable results
	def npForwardTrResult(self, *args, **kwargs):
		trArgs = trGetData(args)
		trKwargs = trGetData(kwargs)
		trResult = self.forward(*trArgs, **trKwargs)
		return trResult

	# Wrapper for passing numpy arrays, converting them to torch arrays, forward network and convert back to numpy
	# @param[in] x The input, which can be a numpy array, or a list/tuple/dict of numpy arrays
	# @return y The output of the network as numpy array
	def npForward(self, *args, **kwargs):
		with tr.no_grad():
			trResult = self.npForwardTrResult(*args, **kwargs)
		npResult = npGetData(trResult)
		return npResult

	def saveWeights(self, path):
		return self.serializer.saveModel(path, stateKeys=["weights", "model_state"])

	def loadWeights(self, path, yolo=False):
		Debug.log(1, "[NWModule::loadWeights] Loading weights from %s" % path)
		if yolo:
			loadedState = NetworkSerializer.readPkl(path)
			assert "weights" in loadedState
			Debug.log(2, "[NWModule::loadWeights] YOLO mode. No state & named params check.")
			self.serializer.doLoadWeights(loadedState["weights"], allowNamedMismatch=True)
		else:
			self.serializer.loadModel(path, stateKeys=["model_state", "weights"])

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
				Debug.log(2, "Warning. Model has unknown state key: %s=%s, possibly added after training. Skipping." \
					% (key, str(self.hyperParameters[key])))
				continue
			loadedState = state[key]
			modelState = self.hyperParameters[key]

			if not deepCheckEqual(loadedState, modelState):
				return False
		return True
	
	def isTrainable(self):
		for param in self.parameters():
			if param.requires_grad == True:
				return True
		return False

	### Getters and setters

	# Returns only the callbacks that are of subclass Callback (not metrics)
	def getCallbacks(self):
		res = list(filter(lambda x : not isBaseOf(x, Metric), self.callbacks.values()))
		return res

	def getMetrics(self):
		res = list(filter(lambda x : isBaseOf(x, Metric), self.callbacks.values()))
		return res

	def getMetric(self, metricName) -> Metric:
		for key, callback in self.callbacks.items():
			if not isBaseOf(callback, Metric):
				continue
			if callback.name == metricName:
				return callback
		assert False, "Metric %s was not found. Use adddMetrics() properly first." % metricName

	def getTrainHistory(self):
		return self.trainHistory

	def setTrainableWeights(self, value):
		for param in self.parameters():
		    param.requires_grad = value

	# Optimizer can be either a class or an object. If it's a class, it will be instantiated on all the trainable
	#  parameters, and using the arguments in the variable kwargs. If it's an object, we will just use that object,
	#  and assume it's correct (for example if we want only some parameters to be trained this has to be used)
	# Examples of usage: model.setOptimizer(nn.Adam, lr=0.01), model.setOptimizer(nn.Adam(model.parameters(), lr=0.01))
	def setOptimizer(self, optimizer, **kwargs):
		if isinstance(optimizer, optim.Optimizer):
			self.optimizer = optimizer
		else:
			trainableParams = list(filter(lambda p : p.requires_grad, self.parameters()))
			assert len(trainableParams) > 0, "Optimizer must have some trainable parameters."
			self.optimizer = optimizer(trainableParams, **kwargs)
		self.optimizer.storedArgs = kwargs

	def getOptimizer(self):
		return self.optimizer

	def getOptimizerStr(self):
		optimizer = self.getOptimizer()
		return _getOptimizerStr(optimizer)

	def setOptimizerScheduler(self, scheduler, **kwargs):
		assert not self.getOptimizer() is None, "Optimizer must be set before scheduler!"
		if isinstance(scheduler, optim.lr_scheduler._LRScheduler):
			self.optimizerScheduler = scheduler
		else:
			self.optimizerScheduler = scheduler(optimizer=self.getOptimizer(), **kwargs)
			# Some schedulers need acces to the model's object. Others, will not have this argument.
			self.optimizerScheduler.model = self
			self.optimizerScheduler.storedArgs = kwargs

	def setCriterion(self, criterion:Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
		self.criterion = criterion
