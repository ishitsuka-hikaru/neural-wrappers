from typing import Union, Tuple, Any
from abc import ABC, abstractmethod
from .callback_name import CallbackName

class Callback(ABC):
	def __init__(self, name:Union[str, Tuple[Any]] = None):
		self.setName(name)

	def setName(self, name:Union[str, Tuple[Any]] = None):
		if name is None:
			name = str(self)
		self.name = CallbackName(name)

	def getName(self) -> CallbackName:
		return self.name

	# This is used by complex MetricAsCallbacks where we do some stateful computation at every iteration and we want
	#  to reduce it gracefully at the end of the epoch, so it can be stored in trainHistory, as well as for other
	#  callbacks to work nicely with it (SaveModels, PlotCallbacks, etc.). So, we apply a reduction function (default
	#  is identity, which might or might not work depending on algorithm).
	def epochReduceFunction(self, results):
		return results

	def iterationReduceFunction(self, results):
		return results

	def onEpochStart(self, **kwargs):
		pass

	def onEpochEnd(self, **kwargs):
		pass

	def onIterationStart(self, **kwargs):
		pass

	def onIterationEnd(self, results, labels, **kwargs):
		pass

	# Some callbacks requires some special/additional tinkering when loading a neural network model from a pickle
	#  binary file (i.e scheduler callbacks must update the optimizer using the new model, rather than the old one).
	#  @param[in] additional Usually is the same as returned by onCallbackSave (default: None)
	def onCallbackLoad(self, additional, **kwargs):
		pass

	# Some callbacks require some special/additional tinkering when saving (such as closing files). It should be noted
	#  that it's safe to close files (or any other side-effect action) because callbacks are deepcopied before this
	#  method is called (in saveModel)
	def onCallbackSave(self, **kwargs):
		pass
