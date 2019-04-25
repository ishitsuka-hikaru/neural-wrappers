# network_serializer.py Script that handles saving/loading a NeuralNetworkPyTorch class (weights, state etc.)
import torch as tr
from copy import deepcopy
from .pytorch_utils import maybeCuda, getNumParams, getOptimizerStr
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../utilities")
from utils import isBaseOf
import neural_wrappers.callbacks
from collections import OrderedDict

class NetworkSerializer:
	# @param[in] The model upon which this serializer works.
	def __init__(self, model):
		self.model = model

	## Saving ##

	# @brief Stores a model (with all its caveats: weights, optimizer, history and callbacks)
	# @param[in] path The path where the serialized object is stored
	def saveModel(self, path, stateKeys):
		assert len(stateKeys) > 0
		state = {}
		for key in stateKeys:
			if key == "weights":
				state[key] = self.doSaveWeights()
			elif key == "optimizer":
				state[key] = self.doSaveOptimizer()
			elif key == "history_dict":
				state[key] = self.doSaveHistoryDict()
			elif key == "callbacks":
				state[key] = self.doSaveCallbacks()
			elif key == "model_state":
				state[key] = self.model.onModelSave()
			else:
				assert False, "Got unknown key %s" % (key)
		tr.save(state, path)

	# @brief Handles saving the weights of the model
	# @return A list of all the parameters (converted to CPU) so they are pickle-able
	def doSaveWeights(self):
		trainableParams = self.model.getTrainableParameters()
		cpuTrainableParams = list(map(lambda x : x.cpu(), trainableParams))
		return cpuTrainableParams

	# @brief Handles saving the optimizer of the model
	def doSaveOptimizer(self):
		assert self.model.optimizer != None, "No optimizer was set for this model. Cannot save."
		optimizerType = type(self.model.optimizer)
		optimizerState = self.model.optimizer.state_dict()
		return {"state" : optimizerState, "type" : optimizerType}

	def doSaveHistoryDict(self):
		return self.model.trainHistory

	def doSaveCallbacks(self):
		callbacksAdditional = []
		callbacks = []
		callbacksOriginalPositions = []
		for i, key in enumerate(self.model.callbacks):
			# Store only callbacks, not MetricAsCallbacks (As they are lambdas which cannot be pickle'd).
			# Metrics must be reloaded anyway, as they do not hold any (global) state, like full Callbacks do.
			callback = self.model.callbacks[key]
			if isBaseOf(callback, neural_wrappers.callbacks.MetricAsCallback):
				callbacksOriginalPositions.append(callback.name)
			else:
				additional = callback.onCallbackSave(model=self.model)
				callbacksAdditional.append(deepcopy(additional))
				callbacks.append(deepcopy(callback))
				callbacksOriginalPositions.append(None)
			# Pretty awkward, but we need to restore the state of this callback (not the one that stored). Calling
			#  onCallbackSave must make the object deep-copyable and pickle-able, but it may leave it in a bad state
			#  (closed files etc.). But we may need to continue using that callback as well (such as storing models
			#  every epoch, but also continuing training), thus we need to "repair" this callback as if we'd load it
			#  from state.
				callback.onCallbackLoad(additional, model=self.model)
		return {"state" : callbacks, "additional" : callbacksAdditional, \
			"callbacks_positions" : callbacksOriginalPositions}

	## Loading ##

	# Loads a stored binary model
	def loadModel(self, path, stateKeys):
		assert len(stateKeys) > 0
		try:
			loadedState = tr.load(path)
		except Exception:
			print("Exception raised while loading model with tr.load(). Forcing CPU load")
			loadedState = tr.load(path, map_location=lambda storage, loc: storage)

		print("Loading model from %s" % (path))
		if not "model_state" in loadedState:
			print("Warning, no model state dictionary for this model (obsolete behaviour). Ignoring.")
			loadedState["model_state"] = None

		if not self.model.onModelLoad(loadedState["model_state"]):
			loaded = loadedState["model_state"]
			current = self.model.onModelSave()
			raise Exception("Could not correclty load the model state loaded: %s vs. current: %s" % (loaded, current))

		for key in stateKeys:
			if key == "weights":
				self.doLoadWeights(loadedState)
			elif key == "optimizer":
				self.doLoadOptimizer(loadedState)
			elif key == "history_dict":
				self.doLoadHistoryDict(loadedState)
			elif key == "callbacks":
				self.doLoadCallbacks(loadedState)
			elif key == "model_state":
				pass
			else:
				assert False, "Got unknown key %s" % (key)
		print("Finished loading model")

	# Handles loading weights from a model.
	def doLoadWeights(self, loadedState):
		if not "weights" in loadedState and "params" in loadedState:
			print("Warning: Depcrecated model, using \"params\" key instead of \"weights\".")
			loadedState["weights"] = loadedState["params"]

		assert "weights" in loadedState
		params = loadedState["weights"]
		loadedParams, _ = getNumParams(params)
		trainableParams = self.model.getTrainableParameters()
		thisParams, _ = getNumParams(trainableParams)
		if loadedParams != thisParams:
			raise Exception("Inconsistent parameters: %d vs %d." % (loadedParams, thisParams))

		for i, item in enumerate(trainableParams):
			if item.shape != params[i].shape:
				raise Exception("Inconsistent parameters: %d vs %d." % (item.shape, params[i].shape))
			with tr.no_grad():
				item[:] = maybeCuda(params[i][:])
			item.requires_grad_(True)
		print("Succesfully loaded weights (%d parameters) " % (loadedParams))

	def doLoadOptimizer(self, loadedState):
		if not "optimizer" in loadedState and ("optimizer_type" in loadedState and "optimizer_state" in loadedState):
			print("Warning: Depcrecated model, using \"optimizer_type\" and \"optimizer_state\" keys" +\
				"instead of \"optimizer\".")
			loadedState["optimizer"] = \
				{"state" : loadedState["optimizer_state"], "type" : loadedState["optimizer_type"]}
		assert "optimizer" in loadedState
	
		# Create a new instance of the optimizer. Some optimizers require a lr to be set as well
		optimizerDict = loadedState["optimizer"]
		self.model.setOptimizer(optimizerDict["type"], lr=0.01)
		self.model.optimizer.load_state_dict(optimizerDict["state"])

		# Optimizer consistency checks
		# Not sure if/how we can use this (not always ordered)
		# l1 = list(model.optimizer.state_dict()["state"].keys())
		trainableParams = self.model.getTrainableParameters()
		l2 = self.model.optimizer.state_dict()["param_groups"][0]["params"]
		l3 = list(map(lambda x : id(x), trainableParams))
		assert l2 == l3, "Something was wrong with loading optimizer"
		print("Succesfully loaded optimizer: %s" % (getOptimizerStr(self.model.optimizer)))

	def doLoadHistoryDict(self, loadedState):
		assert "history_dict" in loadedState
		trainHistory = loadedState["history_dict"]
		self.model.trainHistory = deepcopy(trainHistory)
		self.model.currentEpoch = len(trainHistory) + 1
		print("Succesfully loaded model history (epoch %d)" % (len(trainHistory)))

	def doLoadCallbacks(self, loadedState):
		assert "callbacks" in loadedState
		callbacks = loadedState["callbacks"]["state"]
		additionals = loadedState["callbacks"]["additional"]
		originalPositions = loadedState["callbacks"]["callbacks_positions"]

		filteredPositions = list(filter(lambda x : type(x) is str, originalPositions))
		# This filtering is needed if we're doing save/load on the same model (such as loading and storing very often
		#  so there are some callbacks that need to be reloaded.
		metricCallbacks = {k : v for k, v in self.model.callbacks.items() \
			if isBaseOf(v, neural_wrappers.callbacks.MetricAsCallback) }
		assert len(filteredPositions) == len(metricCallbacks), \
			"Some metrics were saved: %s, but the list of loaded callbacks is different %s" \
			% (filteredPositions, list(metricCallbacks.keys()))

		# Create a new OrederedDict, with the correct order (local metrics + stored callbacks), so we can load the
		#  topological sort correctly.
		newCallbacks = OrderedDict()
		j = 0
		# TODO: might have to reimplement this perhaps because of changing callback names before/after loading
		#  which might mess up topological sort.
		for i in range(len(originalPositions)):
			# Loading stored callbacks with state
			if originalPositions[i] == None:
				key = callbacks[j].name
				value = callbacks[j]
				additional = additionals[j]
				value.onCallbackLoad(additional, model=self.model)
				j += 1
			# Loading stored metrics without state (assumed setMetrics is called identically as it was before storing)
			# This includes setCriterion as well.
			else:
				key = originalPositions[i]
				value = metricCallbacks[key]
			newCallbacks[key] = value
		self.model.callbacks = newCallbacks
		print("Succesfully loaded %d callbacks" % (len(callbacks)))