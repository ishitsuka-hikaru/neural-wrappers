from torch.optim.lr_scheduler import ReduceLROnPlateau as BaseModel
from ..callbacks import CallbackName, Callback
from ..utilities import isBaseOf

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
