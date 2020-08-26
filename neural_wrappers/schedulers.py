from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau as BaseModel
from overrides import overrides
from copy import deepcopy
from .pytorch import NeuralNetworkPyTorch
from .callbacks import CallbackName, Callback
from .utilities import isBaseOf

class ReduceLROnPlateau:
	def __init__(self, **kwargs):
		if isinstance(kwargs["metric"], str):
			kwargs["metric"] = CallbackName(kwargs["metric"])
		if isBaseOf(kwargs["metric"], Callback):
			kwargs["metric"] = kwargs["metric"].getName()
		self.metric = kwargs["metric"]
		del kwargs["metric"]
		self.baseModel = BaseModel(**kwargs)
		self.model = None

	def step(self, epoch=None):
		assert not self.model is None
		history = self.model.trainHistory[-1]
		Key = "Validation" if "Validation" in history and not history["Validation"] is None else "Train"
		metric = history[Key][self.metric]
		self.baseModel.step(metric)

	def state_dict(self):
		stateDict = self.baseModel.state_dict()
		stateDict["metric"] = self.metric
		return stateDict

	def load_state_dict(self, state_dict):
		self.metric = state_dict["metric"]
		del state_dict["metric"]
		self.baseModel.load_state_dict(state_dict)

	def __str__(self):
		return "ReduceLROnPlateau"

	def __getattr__(self, name):
		return {
			"optimizer" : self.baseModel.optimizer,
			"num_bad_epochs" : self.baseModel.num_bad_epochs
		}[name]

class ReduceLRAndBacktrackOnPlateau(_LRScheduler):
	def __init__(self, model:NeuralNetworkPyTorch, metricName:CallbackName, patience:int, factor:float):
		assert patience > 0
		self.model = model
		self.metricName = metricName
		self.patience = patience
		self.factor = factor

		self.lastRelevantWeights = self.model.serializer.doSaveWeights()
		self.lastRelevantOptimizer = self.model.getOptimizer().state_dict()
		self.metric = self.model.getMetric(self.metricName)
		self.numBadInARow = 0
		metricExtremes = self.metric.getExtremes()
		self.lastRelevantValue = metricExtremes["min"] if self.metric.getDirection() == "max" \
			else metricExtremes["max"]
		self.storedArgs = None

	def state_dict(self):
		return {
			"lastRelevantWeights" : self.lastRelevantWeights,
			"metricName" : self.metricName,
			"numBadInARow" : self.numBadInARow,
			"lastRelevantValue" : self.lastRelevantValue,
		}

	def load_state_dict(self, state_dict):
		self.lastRelevantWeights = state_dict["lastRelevantWeights"]
		self.metricName = state_dict["metricName"]
		self.metric = self.model.getMetric(self.metricName)
		self.numBadInARow = state_dict["numBadInARow"]
		self.lastRelevantValue = state_dict["lastRelevantValue"]

	@overrides
	def step(self):
		trainHistory = self.model.trainHistory[-1]
		if "Validation" in trainHistory:
			score = self.model.trainHistory[-1]["Validation"][self.metric.getName()]
		else:
			score = self.model.trainHistory[-1]["Train"][self.metric.getName()]

		if not self.metric.compareFunction(score, self.lastRelevantValue):
			self.numBadInARow += 1
		else:
			self.lastRelevantValue = score
			self.numBadInARow = 0
			self.lastRelevantWeights = deepcopy(self.model.serializer.doSaveWeights())
			self.lastRelevantOptimizer = deepcopy(self.model.getOptimizer().state_dict())

		if self.numBadInARow == self.patience:
			print("[ReduceLRAndBacktrackOnPlateau] Applying reduce lr and backtracking.")
			self.numBadInARow = 0
			self.model.serializer.doLoadWeights(self.lastRelevantWeights)
			self.model.getOptimizer().load_state_dict(self.lastRelevantOptimizer)
			for param_group in self.model.getOptimizer().param_groups:
				oldLR = float(param_group["lr"])
				newLR = oldLR / self.factor
				param_group["lr"] = newLR

	def __str__(self):
		return "ReduceLRAndBacktrackOnPlateau (Patience: %d. Factor: %2.2f)" % (self.patience, self.factor)