import sys
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from copy import deepcopy

from torch.autograd import Variable
from neural_wrappers.transforms import *
from neural_wrappers.metrics import Accuracy, Loss
from neural_wrappers.utilities import makeGenerator, LinePrinter
from neural_wrappers.callbacks import Callback
from .utils import maybeCuda, maybeCpu, getNumParams, getOptimizerStr

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class NeuralNetworkPyTorch(nn.Module):
	def __init__(self):
		self.optimizer = None
		self.criterion = None
		self.metrics = {"Loss" : Loss()}
		self.currentEpoch = 1
		# A list that stores various information about the model at each epoch. The index in the list represents the
		#  epoch value. Each value of the list is a dictionary that holds by default only loss value, but callbacks
		#  can add more items to this (like confusion matrix or accuracy, see mnist example).
		self.trainHistory = []
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

	def populateHistoryDict(self, message, **kwargs):
		assert not kwargs["trainHistory"] is None
		trainHistory = kwargs["trainHistory"]
		trainHistory["duration"] = kwargs["duration"]
		trainHistory["trainMetrics"] = deepcopy(kwargs["trainMetrics"])
		trainHistory["validationMetrics"] = deepcopy(kwargs["validationMetrics"])
		trainHistory["message"] = message

	# Results come in torch format, but callbacks require numpy, so convert the results back to numpy format
	def getNpData(self, results):
		npResults = None
		if type(results) in (list, tuple):
			npResults = []
			for result in results:
				npResult = self.getNpData(result)
				npResults.append(npResult)
		elif type(results) is Variable:
			 npResults = maybeCpu(results.data).numpy()
		return npResults

	# Equivalent of the function above, but using the data from generator (which comes in numpy format)
	def getTrData(self, data, optimize):
		trData = None
		if data is None:
			return data
		elif type(data) in (list, tuple):
			trData = []
			for item in data:
				trItem = self.getTrData(item, optimize)
				trData.append(trItem)
		elif type(data) is np.ndarray:
			trData = Variable(maybeCuda(tr.from_numpy(data)), requires_grad=optimize)
		return trData

	def checkCallbacks(self, callbacks):
		for callback in callbacks:
			mro = type(callback).mro()
			assert Callback in type(callback).mro(), \
				"Expected only subclass of types Callback, got type %s" % (type(callback))

	# Basic method that does a forward phase for one epoch given a generator. It can apply a step of optimizer or not.
	# @param[in] generator Object used to get a batch of data and labels at each step
	# @param[in] stepsPerEpoch How many items to be generated by the generator
	# @param[in] metrics A dictionary containing the metrics over which the epoch is run
	# @param[in] optimize If true, then the optimizer is also called after each iteration
	# @return The mean metrics over all the steps.
	def run_one_epoch(self, generator, stepsPerEpoch, callbacks=[], optimize=False, printMessage=False, debug=False):
		assert "Loss" in self.metrics.keys(), "At least one metric is required and Loss must be in them"
		self.checkCallbacks(callbacks)

		metricResults = {metric : 0 for metric in self.metrics.keys()}
		linePrinter = LinePrinter()
		i = 0
		startTime = datetime.now()

		# Call onEpochStart here, using only basic args
		callbackArgs = {
			"model" : self,
			"epoch" : self.currentEpoch,
			"trainHistory" : self.trainHistory[self.currentEpoch - 1] if self.trainHistory != [] else None
		}
		for callback in callbacks:
			callback.onEpochStart(**callbackArgs)

		# The protocol requires the generator to have 2 items, inputs and labels (both can be None). If there are more
		#  inputs, they can be packed together (stacked) or put into a list, in which case the ntwork will receive the
		#  same list, but every element in the list is tranasformed in torch format.
		for i, items in enumerate(generator):
			npInputs, npLabels = items
			trInputs = self.getTrData(npInputs, optimize=optimize)
			trLabels = self.getTrData(npLabels, optimize=False)

			trResults = self.forward(trInputs)
			npResults = self.getNpData(trResults)
			
			loss = self.criterion(trResults, trLabels)
			npLoss = maybeCpu(loss.data).numpy()
			# if debug:
				# print("\nLoss: %2.6f" % (npLoss))

			if optimize:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			# Iteration callbacks are called here (i.e. for plotting results!)
			callbackArgs = {
				"data" : npInputs,
				"labels" : npLabels,
				"results" : npResults,
				"loss" : npLoss,
				"iteration" : i,
				"numIterations" : stepsPerEpoch
			}
			for callback in callbacks:
				callback.onIterationEnd(**callbackArgs)

			for metric in self.metrics:
				metricResults[metric] += self.metrics[metric](npResults, npLabels, loss=npLoss)

			iterFinishTime = (datetime.now() - startTime)
			if printMessage:
				message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
				for metric in metricResults:
					message += " %s: %2.2f." % (metric, metricResults[metric] / (i + 1))
				# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
				#  the current estimation per epoch. That value minus current time is the current estimation for
				#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
				ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
				message += " ETA: %s" % (ETA)
				linePrinter.print(message)

			if i == stepsPerEpoch - 1:
				break

		if i != stepsPerEpoch - 1:
			sys.stderr.write("Warning! Number of iterations (%d) does not match expected iterations in reader (%d)" % \
				(i, stepsPerEpoch - 1))
		for metric in metricResults:
			metricResults[metric] /= stepsPerEpoch
		return metricResults

	def test_generator(self, generator, stepsPerEpoch, callbacks=[], printMessage=False):
		now = datetime.now()
		resultMetrics = self.run_one_epoch(generator, stepsPerEpoch, callbacks=callbacks, optimize=False, \
			printMessage=printMessage)
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

	def test_model(self, data, labels, batchSize, callbacks=[], printMessage=False):
		dataGenerator = makeGenerator(data, labels, batchSize)
		numIterations = data.shape[0] // batchSize + (data.shape[0] % batchSize != 0)
		return self.test_generator(dataGenerator, stepsPerEpoch=numIterations, callbacks=callbacks, \
			printMessage=printMessage)

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
		for metric in trainMetrics:
			message += " %s: %2.2f." % (metric, trainMetrics[metric])

		if not validationMetrics is None:
			for metric in validationMetrics:
				message += " %s: %2.2f." % ("Val " + metric, validationMetrics[metric])

		message += " Took: %s." % (duration)
		return message

	# @param[in] generator Generator which is used to get items for numEpochs epochs, each taking stepsPerEpoch steps
	# @param[in] stepsPerEpoch How many steps each epoch takes (assumed constant). The generator must generate this
	#  amount of items every epoch.
	# @param[in] numEpochs The number of epochs the network is trained for
	# @param[in] callback A list of callbacks, which must be of types either IterationCallback or EpochCallback. The
	#  first ones are sent to run_one_epoch method, and are called at each iteration (the method onIterationEnd). Both
	#  of the types are called at the en of epoch (method onEpochEnd).
	def train_generator(self, generator, stepsPerEpoch, numEpochs, callbacks=[], validationGenerator=None, \
		validationSteps=0, printMessage=True):
		assert self.optimizer != None and self.criterion != None, "Set optimizer and criterion before training"
		self.checkCallbacks(callbacks)

		if printMessage:
			sys.stdout.write("Training for %d epochs...\n" % (numEpochs))

		# for epoch in range(self.startEpoch, numEpochs + 1):
		while self.currentEpoch < numEpochs + 1:
			# Add this epoch to the trainHistory list, which is used to track history
			self.trainHistory.append({})
			assert len(self.trainHistory) == self.currentEpoch

			# Run for training data and append the results
			now = datetime.now()
			# No iteration callbacks are used if there is a validation set (so iteration callbacks are only
			#  done on validation set). If no validation set is used, the iteration callbacks are used on train set.
			trainCallbacks = [] if validationGenerator != None else callbacks
			trainMetrics = self.run_one_epoch(generator, stepsPerEpoch, callbacks=trainCallbacks, \
				optimize=True, printMessage=printMessage)

			# Run for validation data and append the results
			if validationGenerator != None:
				validationMetrics = self.run_one_epoch(validationGenerator, validationSteps, \
					callbacks=callbacks, optimize=False, printMessage=False)
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

			# Print message is also computed in similar fashion using caallback arguments
			message = self.computePrintMessage(**callbackArgs)
			if printMessage:
				sys.stdout.write(message + "\n")
				sys.stdout.flush()

			# Add basic value to the history dictionary (just loss and time)
			self.populateHistoryDict(message, **callbackArgs)
			for callback in callbacks:
				callback.onEpochEnd(**callbackArgs)

			self.currentEpoch += 1

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
		# The file can be a weights file (just weights then) or a state file (so file["weights"] are the weights)
		if type(params) == dict:
			if not "weights" in params and "params" in params:
				print("Warning: Depcrecated model, using \"params\" key instead of \"weights\".")
				params["weights"] = params["params"]
			self._load_weights(params["weights"])
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
			"weights" : list(map(lambda x : x.cpu(), self.parameters())),
			"optimizer_type" : type(self.optimizer),
			"optimizer_state" : self.optimizer.state_dict(),
			"history_dict" : self.trainHistory
		}
		tr.save(state, path)

	def load_history(self, trainHistory):
		self.trainHistory = deepcopy(trainHistory)
		self.currentEpoch = len(self.trainHistory) + 1

	def _load_model(self, loaded_model):
		if not "weights" in loaded_model and "params" in loaded_model:
			print("Warning: Depcrecated model, using \"params\" key instead of \"weights\".")
			loaded_model["weights"] = loaded_model["params"]
		self._load_weights(loaded_model["weights"])

		# Create a new instance of the optimizer. Some optimizers require a lr to be set as well
		self.setOptimizer(loaded_model["optimizer_type"], lr=0.01)
		self.optimizer.load_state_dict(loaded_model["optimizer_state"])

		# Optimizer consistency checks
		# l1 = list(self.optimizer.state_dict()["state"].keys()) # Not sure if/how we can use this (not always ordered)
		l2 = self.optimizer.state_dict()["param_groups"][0]["params"]
		l3 = list(map(lambda x : id(x), self.parameters()))
		assert l2 == l3, "Something was wrong with loading optimizer"

	def load_model(self, path):
		loaded_model = tr.load(path)
		self._load_model(loaded_model)

		if "history_dict" in loaded_model:
			self.load_history(loaded_model["history_dict"])
			print("Succesfully loaded model (with history, epoch %d)" % (self.currentEpoch))
		else:
			print("Succesfully loaded model (no history)")