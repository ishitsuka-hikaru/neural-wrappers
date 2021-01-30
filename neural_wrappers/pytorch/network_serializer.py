# network_serializer.py Script that handles saving/loading a NWModule class (weights, state etc.)
import torch as tr
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from .utils import getNumParams, getTrainableParameters, _computeNumParams, device
from ..metrics import Metric
from ..utilities import isBaseOf, deepCheckEqual, isPicklable, npCloseEnough, Debug

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
	#  NWModule model: weights, optimizer, history and callbacks state.
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
		def f(optimizer, scheduler):
			assert not optimizer is None, "No optimizer was set for this model. Cannot save."
			optimizerType = type(optimizer)
			optimizerState = optimizer.state_dict()
			optimizerKwargs = optimizer.storedArgs
			Dict = {"state" : optimizerState, "type" : optimizerType, "kwargs" : optimizerKwargs}

			# If there is also an optimizer scheduler appended to this optimizer, save it as well
			if not scheduler is None:
				Dict["scheduler_state"] = scheduler.state_dict()
				Dict["scheduler_type"] = type(scheduler)
				Dict["scheduler_kwargs"] = scheduler.storedArgs
			return Dict

		optimizer = self.model.getOptimizer()
		scheduler = self.model.optimizerScheduler
		res = f(optimizer, scheduler)
		return res

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
			if isBaseOf(callback, Metric):
				callbacksOriginalPositions.append(callback.name)
			else:
				additional = callback.onCallbackSave(model=self.model)
				assert isPicklable(additional), "Callback %s is not pickable" % callback.name
				callbacksAdditional.append(deepcopy(additional))
				assert isPicklable(callback), "Callback %s is not pickable" % callback.name
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
			Debug.log(2, "Exception raised while loading model with tr.load(). Forcing CPU load")
			loadedState = tr.load(path, map_location=lambda storage, loc: storage)
		return loadedState

	# Loads a stored binary model
	def checkModelState(self, loadedState):
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


	def loadModel(self, path, stateKeys):
		assert len(stateKeys) > 0
		loadedState = NetworkSerializer.readPkl(path)

		Debug.log(1, "Loading model from %s" % (path))
		self.checkModelState(loadedState)

		for key in stateKeys:
			if key == "weights":
				assert "weights" in loadedState
				self.doLoadWeights(loadedState["weights"])
				nParams = _computeNumParams(loadedState["weights"])
				Debug.log(2, "Succesfully loaded weights (%d parameters) " % nParams)
			elif key == "optimizer":
				assert "optimizer" in loadedState
				self.doLoadOptimizer(loadedState["optimizer"])
			elif key == "history_dict":
				assert "history_dict" in loadedState
				self.doLoadHistoryDict(loadedState["history_dict"])
			elif key == "callbacks":
				assert "callbacks" in loadedState
				self.doLoadCallbacks(loadedState["callbacks"])
			elif key == "model_state":
				pass
			else:
				assert False, "Got unknown key %s" % (key)
		Debug.log(2, "Finished loading model")

	# Handles loading weights from a model.
	def doLoadWeights(self, loadedParams, allowNamedMismatch=False):
		trainableParams = getTrainableParameters(self.model)
		numTrainableParams = _computeNumParams(trainableParams)
		numLoadedParams = _computeNumParams(loadedParams)
		assert numLoadedParams == numTrainableParams, "Inconsistent parameters: Loaded: %d vs Model (trainable): %d." \
			% (numLoadedParams, numTrainableParams)

		namedTrainableParams = sorted(list(trainableParams.keys()))
		namedLoadedParams = sorted(list(loadedParams.keys()))

		try:
			assert namedTrainableParams == namedLoadedParams, "Old behaviour model not supported anymore."
			for key in namedTrainableParams:
				assert trainableParams[key].shape == loadedParams[key].shape, "This: %s:%s. Loaded:%s" % \
					(nameLoadedParam, str(trainableParam.shape), str(loadedParam.shape))
			newParams = loadedParams
		except Exception as e:
			if not allowNamedMismatch:
				raise Exception(e)
			# This may come in handy at some points when we have renamed/reclassed a model that is already trained.
			newParams = {}
			for param, loadedParam in zip(namedTrainableParams, namedLoadedParams):
				newParams[param] = loadedParams[loadedParam]

		# TODO: Make strict=True and add fake params in the if above (including BN/Dropout).
		missing, unexpected = self.model.load_state_dict(newParams, strict=False)
		if len(missing) > 0:
			Debug.log(2, "Loaded partial model. Missing %d keys (got %d keys)" % (len(missing), len(newParams)))
		if len(unexpected):
			Debug.log(1, "Unexpected %d keys in the loaded model" % (len(unexpected)))

	def doLoadOptimizer(self, optimizerDict):
		assert "kwargs" in optimizerDict
		assert not self.model.getOptimizer() is None, "Set optimizer first before loading the model."
		loadedType = type(self.model.getOptimizer())
		assert optimizerDict["type"] == loadedType, "Optimizers: %s vs %s" % (optimizerDict["type"], loadedType)
		self.model.getOptimizer().load_state_dict(optimizerDict["state"])
		self.model.getOptimizer().storedArgs = optimizerDict["kwargs"]
		Debug.log(2, "Succesfully loaded optimizer: %s" % (self.model.getOptimizerStr()))

		if "scheduler_state" in optimizerDict:
			assert not self.model.optimizerScheduler is None, "Set scheduler first before loading the model."
			loadedSchedulerType = type(self.model.optimizerScheduler)
			assert optimizerDict["scheduler_type"] == loadedSchedulerType, \
				"Schedulers: %s vs %s" % (optimizerDict["scheduler_type"], loadedSchedulerType)
			self.model.optimizerScheduler.load_state_dict(optimizerDict["scheduler_state"])
			self.model.optimizerScheduler.storedArgs = optimizerDict["scheduler_kwargs"]
			Debug.log(2, "Succesfully loaded optimizer scheduler: %s" % (self.model.optimizerScheduler))

	def doLoadHistoryDict(self, trainHistory):
		self.model.trainHistory = deepcopy(trainHistory)
		self.model.currentEpoch = len(trainHistory) + 1
		Debug.log(2, "Succesfully loaded model history (epoch %d)" % (len(trainHistory)))

	def doLoadCallbacks(self, loadedState):
		callbacks = loadedState["state"]
		additionals = loadedState["additional"]
		originalPositions = loadedState["callbacks_positions"]
		topologicalSort = loadedState["topological_sort"]

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
				value = None
				for callback in metricCallbacks:
					if callback.getName() == key:
						value = callback
						break
				assert not value is None, "Couldn't find %s" % (key)
			newCallbacks[key] = value

		self.model.callbacks = newCallbacks
		self.model.topologicalSort = topologicalSort
		self.model.topologicalKeys = np.array(list(self.model.callbacks.keys()))[topologicalSort]

		numMetrics = len(self.model.getMetrics())
		numAll = len(self.model.callbacks)
		Debug.log(2, "Succesfully loaded %d callbacks (%d metrics)" % (numAll, numMetrics))
