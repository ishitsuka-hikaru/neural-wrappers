from .metric import Metric, Number

class MetricWithThreshold(Metric):
	def __call__(self, results : Number, labels : Number, **kwargs):
		assert "threshold" in kwargs
		super().__call__(results, labels, **kwargs)
