class Metric:
	# @param[in] direction Defines the "direction" of the metric, as in if the better value means it is minimized or
	#  maximized. For example, Loss functions (or errors in general) are minimized, thus "min". However, other metrics
	#  such as Accuracy or F1Score are to be maximized, hence "max". Defaults to "min".
	def __init__(self, direction="min"):
		assert direction in ("min", "max")
		self.direction = direction

	def __call__(self, results, labels, **kwargs):
		raise NotImplementedError("Should have implemented this")