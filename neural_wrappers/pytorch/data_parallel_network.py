from .network import NeuralNetworkPyTorch
import torch.nn as nn

# Hackish solution so we can use NeuralNetworkPyTorch's methods (run_one_epoch, train_generator etc.) while still
#  using pytorch's DataParallel module, which scatters and gathers data in multiple devices automagically.
class DataParallelNetwork(NeuralNetworkPyTorch):
	def __init__(self, model):
		super().__init__(hyperParameters=model.hyperParameters)
		self.baseModel = nn.DataParallel(model)

	def forward(self, *args, **kwargs):
		return self.baseModel.forward(*args, **kwargs)

	def __str__(self):
		return "General parallel neural network architecture. Update __str__ in your model for more details when " + \
			"using summary."
