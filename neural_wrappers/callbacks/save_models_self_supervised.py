from .save_models import SaveModels

# Used to save self-supervised models.
class SaveModelsSelfSupervised(SaveModels):
	def __init__(self, type="all", **kwargs):
		super().__init__(**kwargs)
		self.name = "SaveModelsSelfSupervised"

	def onEpochEnd(self, **kwargs):
		model = deepcopy(kwargs["model"]).cpu()
		model.setPretrainMode(False)
		kwargs["model"] = model
		super().onEpochEnd(**kwargs)