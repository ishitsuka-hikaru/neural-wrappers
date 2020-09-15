from .nw_module import NWModule

# Wrapper on top of the PyTorch model. Added methods for saving and loading a state. To completly implement a PyTorch
#  model, one must define layers in the object's constructor, call setOptimizer, setCriterion and implement the
#  forward method identically like a normal PyTorch model.
class FeedForwardNetwork(NWModule):
	def __init__(self, hyperParameters={}):
		super().__init__(hyperParameters)
