# network_serializer.py Script that handles saving/loading a NeuralNetworkPyTorch class (weights, state etc.)
import torch as tr
from copy import deepcopy
from .utils import maybeCuda, getNumParams, getOptimizerStr

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
		return list(map(lambda x : x.cpu(), self.model.parameters()))

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
		for callback in self.model.callbacks:
			additional = callback.onCallbackSave(model=self.model)
			callbacksAdditional.append(deepcopy(additional))
			callbacks.append(deepcopy(callback))
			# Pretty awkward, but we need to restore the state of this callback (not the one that stored). Calling
			#  onCallbackSave must make the object deep-copyable and pickle-able, but it may leave it in a bad state
			#  (closed files etc.). But we may need to continue using that callback as well (such as storing models
			#  every epoch, but also continuing training), thus we need to "repair" this callback as if we'd load it
			#  from state.
			callback.onCallbackLoad(additional, model=self.model)
		return {"state" : callbacks, "additional" : callbacksAdditional}

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
		assert self.model.onModelLoad(loadedState["model_state"]) == True

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
		thisParams, _ = getNumParams(self.model.parameters())
		if loadedParams != thisParams:
			raise Exception("Inconsistent parameters: %d vs %d." % (loadedParams, thisParams))

		for i, item in enumerate(self.model.parameters()):
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
		l2 = self.model.optimizer.state_dict()["param_groups"][0]["params"]
		l3 = list(map(lambda x : id(x), self.model.parameters()))
		assert l2 == l3, "Something was wrong with loading optimizer"
		print("Succesfully loaded optimizer: %s" % (getOptimizerStr(self.model.optimizer)))

	def doLoadHistoryDict(self, loadedState):
		assert "history_dict" in loadedState
		trainHistory = loadedState["history_dict"]
		self.model.trainHistory = deepcopy(trainHistory)
		self.model.currentEpoch = len(trainHistory) + 1
		print("Succesfully loaded model history (epoch %d)" % (self.model.currentEpoch))

	def doLoadCallbacks(self, loadedState):
		assert "callbacks" in loadedState
		callbacks = loadedState["callbacks"]["state"]
		additionals = loadedState["callbacks"]["additional"]
		self.model.callbacks = callbacks
		for i in range(len(self.model.callbacks)):
			callback = self.model.callbacks[i]
			additional = additionals[i]
			callback.onCallbackLoad(additional, model=self.model)
		print("Succesfully loaded %d callbacks" % (len(callbacks)))