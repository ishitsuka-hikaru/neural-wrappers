from .nw_module import NWModule
from .utils import npGetData, trGetData
from overrides import overrides

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class FeedForwardNetwork(NWModule):
	def __init__(self, hyperParameters={}):
		super().__init__(hyperParameters)

	@overrides
	def networkAlgorithm(self, trInputs, trLabels, isTraining, isOptimizing):
		assert not self.criterion is None, "Set criterion before training or testing"
		trResults = self.forward(trInputs)
		trLoss = self.criterion(trResults, trLabels)
		self.updateOptimizer(trLoss, isTraining, isOptimizing)
		return trResults, trLoss
