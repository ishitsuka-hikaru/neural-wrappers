from abc import ABC, abstractmethod
from typing import Union, Optional, Callable
from overrides import overrides
from ..utilities import NWNumber
from ..callbacks import Callback

# @brief Base Class for all metrics. It defines a direction, which represents whether the metric is minimized or
#  maximized.
class Metric(Callback):
	# @param[in] direction Defines the "direction" of the metric, as in if the better value means it is minimized or
	#  maximized. For example, Loss functions (or errors in general) are minimized, thus "min". However, other metrics
	#  such as Accuracy or F1Score are to be maximized, hence "max". Defaults to "min".
	def __init__(self, direction : str="min"):
		super().__init__()
		assert direction in ("min", "max")
		self.direction = direction

	# @brief Getter for the direction of the metric
	# @return The direction of the metric
	def getDirection(self) -> str:
		return self.direction

	# @brief The default value of the metric, used by some implementations to define defaults for various statistics
	# @return A value that represents the default value of the metric
	def defaultValue(self) -> NWNumber:
		return 0

	# @brief Provides a sane way of comparing two results of this metric
	# @return Returns a callback that can compare two results and returns a bool value (usually if result a is better
	#  than result b, given the metrics semantic, but can be overriden)
	def compareFunction(self, a:NWNumber, b:NWNumber) -> bool:
		return {
			"min" : lambda a, b : a > b,
			"max" : lambda a, b : a < b,
		}[self.direction]

	@overrides
	def onEpochStart(self, **kwargs):
		pass

	@overrides
	def onEpochEnd(self, **kwargs):
		pass

	@overrides
	def onIterationStart(self, **kwargs):
		pass

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		return self.__call__(results, labels, **kwargs)

	@overrides
	def onCallbackLoad(self, additional, **kwargs):
		pass

	@overrides
	def onCallbackSave(self, **kwargs):
		pass

	# @brief The main method that must be implemented by a metric
	@abstractmethod
	def __call__(self, results : NWNumber, labels : NWNumber, **kwargs):
		pass
