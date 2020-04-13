# network_serializer.py Script that handles saving/loading a NeuralNetworkPyTorch class (weights, state etc.)
import torch as tr
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from .utils import getNumParams, getOptimizerStr, getTrainableParameters, _computeNumParams, device
from ..utilities import isBaseOf, deepCheckEqual, isPicklable, npCloseEnough
from ..callbacks import MetricAsCallback

class NetworkSerializer:
	# @param[in] The model upon which this serializer works.
	def __init__(self, model):
		self.model = model

	## Saving ##

	# @brief Stores a model (with all its caveats: weights, optimizer, history and callbacks)
	# @param[in] path The path where the serialized object is stored
	def saveModel(self, path, stateKeys):
		state = self.doSerialization(stateKeys)
		tr.save(state, path)

	# @brief Computes a serialized version of the model, by storing the state of all caveats that makes up a
	#  NeuralNetworkPyTorch model: weights, optimizer, history and callbacks state.
	# @param[in] stateKeys A list of all keys that are to be stored (saveWeights just stores weights for example)
	# @return returns a serialized version of the model
	def doSerialization(self, stateKeys):
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
			assert isPicklable(state[key]), "Key %s is not pickable" % (key)
		return state

	# @brief Handles saving the weights of the model
	# @return A list of all the parameters (converted to CPU) so they are pickle-able
	def doSaveWeights(self):
		trainableParams = getTrainableParameters(self.model)
		parametersState = {x : self.model.state_dict()[x] for x in sorted(list(trainableParams.keys()))}
		return parametersState

	# @brief Handles saving the optimizer of the model
	def doSaveOptimizer(self):
		assert not self.model.optimizer is None, "No optimizer was set for this model. Cannot save."
		optimizerType = type(self.model.optimizer)
		optimizerState = self.model.optimizer.state_dict()
		optimizerKwargs = self.model.optimizer.storedArgs
		Dict = {"state" : optimizerState, "type" : optimizerType, "kwargs" : optimizerKwargs}

		# If there is also an optimizer scheduler appended to this optimizer, save it as well
		if not self.model.optimizerScheduler is None:
			Dict["scheduler_state"] = self.model.optimizerScheduler.state_dict()
			Dict["scheduler_type"] = type(self.model.optimizerScheduler)
			Dict["scheduler_kwargs"] = self.model.optimizerScheduler.storedArgs

		return Dict

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
			assert key == self.model.callbacks[key].name, ("Some inconsistency between the metric's key and the" + \
				" MetricAsCallback's name was found. Key: %s. Name: %s" % (key, callback.name))
			if isBaseOf(callback, MetricAsCallback):
				callbacksOriginalPositions.append(callback.name)
			else:
				additional = callback.onCallbackSave(model=self.model)
				callbacksAdditional.append(deepcopy(additional))
				callbacks.append(deepcopy(callback))
				callbacksOriginalPositions.append(None)
				# Pretty awkward, but we need to restore the state of this callback (not the one that stored). Calling
				#  onCallbackSave must make the object deep-copyable and pickle-able, but it may leave it in a bad
				#  state (closed files etc.). But we may need to continue using that callback as well (such as
				#  storing models every epoch, but also continuing training), thus we need to "repair" this callback
				#  as if we'd load it from state.
				callback.onCallbackLoad(additional, model=self.model)
		return {"state" : callbacks, "additional" : callbacksAdditional, \
			"callbacks_positions" : callbacksOriginalPositions, "topological_sort" : self.model.topologicalSort}

	## Loading ##

	@staticmethod
	def readPkl(path):
		try:
			loadedState = tr.load(path)
		except Exception:
			print("Exception raised while loading model with tr.load(). Forcing CPU load")
			loadedState = tr.load(path, map_location=lambda storage, loc: storage)
		return loadedState

	# Loads a stored binary model
	def loadModel(self, path, stateKeys):
		assert len(stateKeys) > 0
		loadedState = NetworkSerializer.readPkl(path)

		print("Loading model from %s" % (path))
		if not "model_state" in loadedState:
			print("Warning, no model state dictionary for this model (obsolete behaviour). Ignoring.")
			loadedState["model_state"] = None

		if not self.model.onModelLoad(loadedState["model_state"]):
			loaded = loadedState["model_state"]
			current = self.model.onModelSave()
			Str = "Could not correclty load the model state. \n"
			Str += "- Loaded: %s\n" % (loaded)
			Str += "- Current: %s\n" % (current)
			Str += "- Diffs:\n"
			for key in set(list(loaded.keys()) + list(current.keys())):
				if not key in current:
					Str += "\t- Key '%s' in loaded model, not in current\n" % (key)
					continue
				if not key in loaded:
					Str += "\t- Key '%s' in current model, not in loaded\n"% (key)
					continue
				if(not deepCheckEqual(current[key], loaded[key])):
					Str += "\t- Key '%s' is different:\n" % (str(key))
					Str += "\t\t- current=%s.\n" % (str(current[key]))
					Str += "\t\t- loaded=%s.\n" % (str(loaded[key]))
			raise Exception(Str)

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

	def doLoadWeightsOld(self, loadedState):
		assert "weights" in loadedState
		loadedParams = loadedState["weights"]
		# trainableParams = getTrainableParameters(self.model)
		trainableParams = dict(filter(lambda x : x[1].requires_grad, self.model.named_parameters()))
		numTrainableParams = _computeNumParams(trainableParams)
		numLoadedParams = _computeNumParams({i : x for i, x in zip(range(len(loadedParams)), loadedParams)})

		assert numLoadedParams == numTrainableParams, "Inconsistent parameters: Loaded: %d vs Model (trainable): %d." \
			% (numLoadedParams, numTrainableParams)
		for i, name in enumerate(trainableParams.keys()):
			thisItem = trainableParams[name]
			loadedItem = loadedParams[i]
			if thisItem.shape != loadedItem.shape:
				raise Exception("Inconsistent parameters: %d vs %d." % (thisItem.shape, loadedItem.shape))
			with tr.no_grad():
				thisItem[:] = loadedItem[:].to(device)
			thisItem.requires_grad_(True)
		print("Succesfully loaded weights (%d parameters) " % (numLoadedParams))

	def doLoadWeightsOld2(self, namedTrainableParams, namedLoadedParams, trainableParams, loadedParams):
		print("Loading in old way, where we hope/assume that the order of the keys is identical to the loaded one")
		# Combines names of trainable params with weights from loaded (should apply to all cases) if the loop down
		#  works (Potential bug: RARE CASE WHERE DICT ORDER IS DIFFERENT BUT SAME # OF PARAMS)
		newParams = {}
		for i in range(len(namedTrainableParams)):
			nameTrainableParam = namedTrainableParams[i]
			nameLoadedParam = namedLoadedParams[i]
			trainableParam = trainableParams[nameTrainableParam]
			loadedParam = loadedParams[nameLoadedParam]
			assert trainableParam.shape == loadedParam.shape, "This: %s:%s. Loaded:%s" % \
				(nameLoadedParam, str(trainableParam.shape), str(loadedParam.shape))
			newParams[nameTrainableParam] = loadedParam
		return newParams

	# Handles loading weights from a model.
	def doLoadWeights(self, loadedState):
		assert "weights" in loadedState
		loadedParams = loadedState["weights"]
		if type(loadedParams) == list:
			print("Warning! Model was stored with older version (list instead of named parameters dict). " + \
				"Will attempt to load weights, however having BatchNorm/Dropout will result in .eval() not " + \
				"working as well as other possible bugs.")
			self.doLoadWeightsOld(loadedState)
			return

		trainableParams = getTrainableParameters(self.model)
		numTrainableParams = _computeNumParams(trainableParams)
		numLoadedParams = _computeNumParams(loadedParams)
		assert numLoadedParams == numTrainableParams, "Inconsistent parameters: Loaded: %d vs Model (trainable): %d." \
			% (numLoadedParams, numTrainableParams)

		namedTrainableParams = sorted(list(trainableParams.keys()))
		namedLoadedParams = sorted(list(loadedParams.keys()))
		# Loading in natural way, because keys are identical
		if namedTrainableParams == namedLoadedParams:
			for key in namedTrainableParams:
				assert trainableParams[key].shape == loadedParams[key].shape, "This: %s:%s. Loaded:%s" % \
					(nameLoadedParam, str(trainableParam.shape), str(loadedParam.shape))
			newParams = loadedParams
		else:
			newParams = self.doLoadWeightsOld2(namedTrainableParams, namedLoadedParams, trainableParams, loadedParams)
		self.model.load_state_dict(newParams, strict=True)
		print("Succesfully loaded weights (%d parameters) " % (numLoadedParams))

	def doLoadOptimizer(self, loadedState):
		assert "optimizer" in loadedState
	
		# Create a new instance of the optimizer. Some optimizers require a lr to be set as well
		optimizerDict = loadedState["optimizer"]

		if not "kwargs" in optimizerDict:
			print("Warning: Depcrecated model. No kwargs in optimizerDict. Defaulting to lr=0.01")
			optimizerDict["kwargs"] = {"lr" : 0.01}
		self.model.setOptimizer(optimizerDict["type"], **optimizerDict["kwargs"])
		self.model.optimizer.load_state_dict(optimizerDict["state"])
		self.model.optimizer.storedArgs = optimizerDict["kwargs"]
		print("Succesfully loaded optimizer: %s" % (getOptimizerStr(self.model.optimizer)))

		if "scheduler_state" in optimizerDict:
			self.model.setOptimizerScheduler(optimizerDict["scheduler_type"], **optimizerDict["scheduler_kwargs"])
			self.model.optimizerScheduler.load_state_dict(optimizerDict["scheduler_state"])
			self.model.optimizerScheduler.storedArgs = optimizerDict["scheduler_kwargs"]
			print("Succesfully loaded optimizer scheduler: %s" % (self.model.optimizerScheduler))

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
		topologicalSort = loadedState["callbacks"]["topological_sort"]

		# This filtering is needed if we're doing save/load on the same model (such as loading and storing very often
		#  so there are some callbacks that need to be reloaded.
		metricCallbacks = self.model.getMetrics()
		assert len(metricCallbacks) == len(metricCallbacks), \
			"Some metrics were saved: %s, but the list of loaded callbacks is different %s" \
			% (metricCallbacks, list(metricCallbacks.keys()))

		# Create a new OrederedDict, with the correct order (local metrics + stored callbacks), so we can load the
		#  topological sort correctly.
		newCallbacks = OrderedDict()
		j = 0
		for i in range(len(originalPositions)):
			# Loading stored callbacks with state
			if originalPositions[i] == None:
				key = callbacks[j].name
				value = callbacks[j]
				additional = additionals[j]

				# This should be safe (trainHistory is not empty) because doLoadHistory is called before this method
				kwargs = {
					"model" : self.model,
					"trainHistory" : self.model.trainHistory
				}
				value.onCallbackLoad(additional, **kwargs)
				j += 1
			# Loading stored metrics without state (assumed setMetrics is called identically as it was before storing)
			# This includes setCriterion as well.
			else:
				key = originalPositions[i]
				value = metricCallbacks[key]
			newCallbacks[key] = value

		self.model.callbacks = newCallbacks
		self.model.topologicalSort = topologicalSort
		self.model.topologicalKeys = np.array(list(self.model.callbacks.keys()))[topologicalSort]

		numMetrics = len(self.model.getMetrics())
		numAll = len(self.model.callbacks)
		print("Succesfully loaded %d callbacks (%d metrics)" % (numAll, numMetrics))
