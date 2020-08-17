import numpy as np
# from .metric_as_callback import MetricAsCallback
from ..metrics import MetricWithThreshold
from ..utilities import NWNumber, NWDict, NWSequence, isType
from typing import List, Union, Dict
from collections import OrderedDict

# This wraps a metric with threshold by providing (a variable numober of) thresholds to be tested against
# Results are stored at each iteration and the epoch results are saved as the running mean of results.
class MetricWithThresholdWrapper:
	pass
# class MetricWithThresholdWrapper(MetricAsCallback):
	# def __init__(self, metricName : str, metric : MetricWithThreshold, \
	# 	thresholds : Union[NWNumber, np.ndarray, List[NWNumber]] ):
	# 	super().__init__(metricName, metric)
	# 	self.thresholds = thresholds

	# def defaultValue(self) -> Union[float, int, List[int], NWDict]: #type: ignore[override]
	# 	if isType(self.thresholds, float) or isType(self.thresholds, int): #type: ignore
	# 		return 0
	# 	elif isType(self.thresholds, NWDict): #type: ignore
	# 		return {k : 0 for k in self.thresholds} #type: ignore
	# 	elif isType(self.thresholds, NWSequence): #type: ignore
	# 		return [0 for x in self.thresholds] #type: ignore
	# 	assert False

	# def onIterationEnd(self, results, labels, **kwargs):
	# 	if isType(self.thresholds, float) or isType(self.thresholds, int): #type: ignore
	# 		return self.metric(results, labels, threshold=self.thresholds, **kwargs)
	# 	elif isType(self.thresholds, NWDict): #type: ignore
	# 		return {k : self.metric(results, labels, threshold=self.thresholds[k], **kwargs) for k in self.thresholds}
	# 	elif isType(self.thresholds, NWSequence): #type: ignore
	# 		return np.array([self.metric(results, labels, threshold=x, **kwargs) for x in self.thresholds])