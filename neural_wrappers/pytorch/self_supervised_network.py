from .network import NeuralNetworkPyTorch

class SelfSupervisedNetwork(NeuralNetworkPyTorch):
	def __init__(self, baseModel):
		super().__init__()
		self.baseModel = baseModel
		self.setPretrainMode(True)

	def setPretrainMode(self, mode):
		self.linePrinter.print("Setting pretraining mode to %s. Resetting epoch and train history.\n" % (mode))
		self.currentEpoch = 1
		self.trainHistory = []
		self.pretrain = mode
		if mode == True:
			self.forwardCallback = lambda x : self.pretrain_forward(x)
		else:
			self.forwardCallback = lambda x : self.baseModel.forward(x)
		self.pretrainLayersSetup()

	# @brief Wrapper on top of the model.forward network. In case we are in pre-training phase, use the
	#  pretrain_forward method, which must be implemented by the pretraining model and this can inclde changes in
	#  the network architecture (like changing last layer from classification to reconstruction, for example). If
	#  the pretrain mode is False, we are using the regular base model forward method.
	def forward(self, x):
		return self.forwardCallback(x)

	def pretrain_forward(self, x):
		raise NotImplementedError("Should have implemented this")

	# @brief Method used by all the Self Supervised Networks in order to set up the additional layers needed for the
	#  self supervised task. This method can be a simple "pass", if the same layers are used, just the criterion
	#  is changed. Usually, Only the last layer is changed (from classification to reconstruction FC layer). Because
	#  the last layer may be very large, it is better if we delete that last layer once we no longer need it (set it
	#  to None after loading the pretrained weights). This way, when saving the weights for the real task, the
	#  reconstruction FC layer won't be saved anymore and the model will be loadable from the baseModel only.
	def pretrainLayersSetup(self):
		raise NotImplementedError("Should have implemented this")
