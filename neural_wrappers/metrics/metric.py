from typing import Union, Optional
from ..utilities import NWNumber

# @brief Base Class for all metrics. It defines a direction, which represents whether the metric is minimized or
#  maximized.
class Metric:
	# @param[in] direction Defines the "direction" of the metric, as in if the better value means it is minimized or
	#  maximized. For example, Loss functions (or errors in general) are minimized, thus "min". However, other metrics
	#  such as Accuracy or F1Score are to be maximized, hence "max". Defaults to "min".
	def __init__(self, direction : str="min"):
		assert direction in ("min", "max")
		self.direction = direction

	# @brief Getter for the direction of the metric
	# @return The direction of the metric
	def getDirection(self) -> str:
		return self.direction

	# @brief The reduce function, used by complex callbacks to transform at epoch and a callback into a metric that
	#  can be stored and used safely by other callbacks (i.e. SaveModels or PlotMetrics).
	def epochReduceFunction(self, results) -> NWNumber:
		return results

	def iterationReduceFunction(self, results) -> NWNumber:
		return results

	def defaultValue(self) -> NWNumber:
		return 0

	# @brief The main method that must be implemented by a metric
	def __call__(self, results : NWNumber, labels : NWNumber, **kwargs):
		raise NotImplementedError("Should have implemented this")