from .network import NeuralNetworkPyTorch

class SelfSupervisedNetwork(NeuralNetworkPyTorch):
	def __init__(self, baseModel):
		super().__init__()
		self.baseModel = baseModel
		self.pretrain = True

	def setPretrainMode(self, mode):
		self.linePrinter.print("Setting pretraining mode to %s. Resetting epoch and train history.\n" % (mode))
		self.currentEpoch = 1
		self.trainHistory = []
		self.pretrain = mode

	# @brief Wrapper on top of the model.forward network. In case we are in pre-training phase, use the
	#  pretrain_forward method, which must be implemented by the pretraining model and this can inclde changes in
	#  the network architecture (like changing last layer from classification to reconstruction, for example). If
	#  the pretrain mode is False, we are using the regular base model forward method.
	def forward(self, x):
		if self.pretrain:
			return self.pretrain_forward(x)
		else:
			return self.baseModel.forward(x)

	def pretrain_forward(self, x):
		raise NotImplementedError("Should have implemented this")