import numpy as np
import matplotlib.pyplot as plt
from overrides import overrides
from typing import List, Union, Tuple, Optional
from .callback import Callback
from .callback_name import CallbackName
from ..utilities import deepDictGet


def plotModelMetricHistory(model, metricName, trainHistory):
	metric = model.getMetric(metricName.name)
	numEpochs = len(trainHistory)
	hasValidation = "Validation" in trainHistory[0]

	trainScores = [deepDictGet(epochHistory["Train"], metricName.name) for epochHistory in trainHistory]
	trainScores = [x.mean() for x in trainScores] if isinstance(trainScores[0], np.ndarray) else trainScores
	trainScores = np.array(trainScores)
	valScores = None
	if hasValidation:
		valScores = [deepDictGet(epochHistory["Validation"], metricName.name) for epochHistory in trainHistory]
		valScores = [x.mean() for x in valScores] if isinstance(valScores[0], np.ndarray) else valScores
		valScores = np.array(valScores)

	X = np.arange(1, numEpochs + 1)
	plt.gcf().clf()
	plt.gca().cla()
	plt.plot(X, trainScores, label="Train %s" % (metricName))

	if hasValidation:
		plt.plot(X, valScores, label="Val %s" % (metricName))
		usedValues = valScores
	else:
		usedValues = trainScores

	# Against NaNs killing the training for low data count.
	allValues = np.concatenate([usedValues, trainScores])
	Median = np.median(allValues[np.isfinite(allValues)])
	trainScores[~np.isfinite(trainScores)] = Median
	usedValues[~np.isfinite(usedValues)] = Median
	allValues[~np.isfinite(allValues)] = Median

	# assert plotBestBullet in ("none", "min", "max"), "%s" % plotBestBullet
	if metric.getDirection() == "min":
		minX, minValue = np.argmin(usedValues), np.min(usedValues)
		plt.annotate("Epoch %d\nMin %2.2f" % (minX + 1, minValue), xy=(minX + 1, minValue))
		plt.plot([minX + 1], [minValue], "o")
	elif metric.getDirection() == "max":
		maxX, maxValue = np.argmax(usedValues), np.max(usedValues)
		plt.annotate("Epoch %d\nMax %2.2f" % (maxX + 1, maxValue), xy=(maxX + 1, maxValue))
		plt.plot([maxX + 1], [maxValue], "o")

	# Set the y axis to have some space above and below the plot min/max values so it looks prettier.
	minValue, maxValue = np.min(allValues), np.max(allValues)
	diff = (maxValue - minValue) / 10
	plt.gca().set_ylim(minValue - diff, maxValue + diff)

	# Finally, save the figure with the name of the metric
	plt.xlabel("Epoch")
	plt.ylabel(metricName)
	plt.legend()
	plt.savefig("%s.png" % (metricName), dpi=120)

class PlotMetrics(Callback):
	def __init__(self, metricNames:List[CallbackName], **kwargs):
		assert len(metricNames) > 0, "Expected a list of at least one metric which will be plotted."
		self.metricNames = []
		for metricName in metricNames:
			if isinstance(metricName, Callback):
				metricName = metricName.getName()
			metricName = CallbackName(metricName)
			self.metricNames.append(metricName)
		super().__init__(**kwargs)

	@overrides
	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		model = kwargs["model"]

		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for i in range(len(self.metricNames)):
			plotModelMetricHistory(model, self.metricNames[i], trainHistory)

	@overrides
	def onEpochStart(self, **kwargs):
		pass

	@overrides
	def onIterationStart(self, **kwargs):
		pass

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		pass

	@overrides
	def onCallbackSave(self, model):
		self.directions = None

	@overrides
	def onCallbackLoad(self, additional, **kwargs):
		pass

	@overrides
	def __str__(self):
		assert len(self.metricNames) >= 1
		Str = str(self.metricNames[0])
		for i in range(len(self.metricNames)):
			Str += ", %s" % (str(self.metricNames[i]))
		return "PlotMetrics (%s)" % (Str)