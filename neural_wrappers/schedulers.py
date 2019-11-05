from torch.optim.lr_scheduler import ReduceLROnPlateau as BaseModel

class ReduceLROnPlateau:
	def __init__(self, **kwargs):
		kwargs["metric"] = "Loss" if not "metric" in kwargs else kwargs["metric"]
		self.metric = kwargs["metric"]
		del kwargs["metric"]
		self.baseModel = BaseModel(**kwargs)
		self.model = None

	def step(self, epoch=None):
		assert not self.model is None
		Key = "Validation" if "Validation" in self.model.trainHistory[-1] else "Train"
		metric = self.model.trainHistory[-1][Key][self.metric]
		epoch = self.model.currentEpoch
		self.baseModel.step(metric, epoch)

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
