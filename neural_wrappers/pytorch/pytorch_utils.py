import matplotlib.pyplot as plt
import torch as tr
import numpy as np
import sys
from collections import OrderedDict

class StorePrevState:
	def __init__(self, moduleObj):
		self.moduleObj = moduleObj

	def __enter__(self):
		self.prevState = self.moduleObj.train if self.moduleObj.training else self.moduleObj.eval

	def __exit__(self, type, value, traceback):
		self.prevState()

def maybeCuda(x):
	return x.cuda() if tr.cuda.is_available() and hasattr(x, "cuda") else x

def maybeCpu(x):
	return x.cpu() if tr.cuda.is_available() and hasattr(x, "cpu") else x

def getNumParams(params):
	numParams, numTrainable = 0, 0
	for param in params:
		npParamCount = np.prod(param.data.shape)
		numParams += npParamCount
		if param.requires_grad:
			numTrainable += npParamCount
	return numParams, numTrainable

def getOptimizerStr(optimizer):
	if optimizer is None:
		return "None"

	groups = optimizer.param_groups[0]
	if type(optimizer) == tr.optim.SGD:
		optimizerType = "SGD"
		params = "Learning rate: %s, Momentum: %s, Dampening: %s, Weight Decay: %s, Nesterov: %s" % (groups["lr"], \
			groups["momentum"], groups["dampening"], groups["weight_decay"], groups["nesterov"])
	elif type(optimizer) == tr.optim.Adam:
		optimizerType = "Adam"
		params = "Learning rate: %s, Betas: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], groups["betas"], \
			groups["eps"], groups["weight_decay"])
	else:
		raise NotImplementedError("Not yet implemneted optimizer str for %s" % (type(optimizer)))
	return "%s. %s" % (optimizerType, params)

# Results come in torch format, but callbacks require numpy, so convert the results back to numpy format
def getNpData(results):
	npResults = None
	if results is None:
		return None

	if type(results) in (list, tuple):
		npResults = []
		for result in results:
			npResult = getNpData(result)
			npResults.append(npResult)
	elif type(results) in (dict, OrderedDict):
		npResults = {}
		for key in results:
			npResults[key] = getNpData(results[key])

	elif type(results) == tr.Tensor:
		 npResults = maybeCpu(results.detach()).numpy()
	else:
		assert False, "Got type %s" % (type(results))
	return npResults

# Equivalent of the function above, but using the data from generator (which comes in numpy format)
def getTrData(data):
	trData = None
	if data is None:
		return None

	elif type(data) in (list, tuple):
		trData = []
		for item in data:
			trItem = getTrData(item)
			trData.append(trItem)
	elif type(data) in (dict, OrderedDict):
		trData = {}
		for key in data:
			trData[key] = getTrData(data[key])
	elif type(data) is np.ndarray:
		trData = maybeCuda(tr.from_numpy(data))
	elif type(data) is tr.Tensor:
		trData = maybeCuda(data)
	return trData

def plotModelHistory(model, metric, plotBestBullet, dpi):
	trainHistory = model.trainHistory
	assert metric in trainHistory[0]["trainMetrics"], "Metric %s not found in trainHistory, " + \
		"use setMetrics accordingly"

	# Aggregate all the values from trainHistory into a list and plot them
	trainValues, valValues = [], []
	for epoch in range(len(trainHistory)):
		trainValues.append(trainHistory[epoch]["trainMetrics"][metric])
		if "validationMetrics" in trainHistory[epoch] and trainHistory[epoch]["validationMetrics"]:
			valValues.append(trainHistory[epoch]["validationMetrics"][metric])
	x = np.arange(len(trainValues)) + 1
	plt.gcf().clf()
	plt.gca().cla()
	plt.plot(x, trainValues, label="Train %s" % (metric))

	# If we don't have a validation results, further analysis will be done on training results
	if "validationMetrics" in trainHistory[0] and trainHistory[0]["validationMetrics"]:
		plt.plot(x, valValues, label="Val %s" % (metric))
		usedValues = valValues
	else:
		usedValues = trainValues
	plt.legend()

	# Here, we put a bullet on the best epoch (which can be min for loss, max for accuracy or none for neither)
	if plotBestBullet == "none":
		pass
	elif plotBestBullet == "min":
		minX, minValue = np.argmin(usedValues), np.min(usedValues)
		offset = minValue // 2
		plt.annotate("Epoch %d\nMin %2.2f" % (minX + 1, minValue), xy=(minX + 1, minValue))
		plt.plot([minX + 1], [minValue], "o")
	elif plotBestBullet == "max":
		maxX, maxValue = np.argmax(usedValues), np.max(usedValues)
		offset = maxValue // 2
		plt.annotate("Epoch %d\nMax %2.2f" % (maxX + 1, maxValue), xy=(maxX + 1, maxValue))
		plt.plot([maxX + 1], [maxValue], "o")
	else:
		assert False, "Expected: \"min\", \"max\" or \"none\""

	# Set the y axis to have some space above and below the plot min/max values so it looks prettier.
	minValue = min(np.min(usedValues), np.min(trainValues))
	maxValue = max(np.max(usedValues), np.max(trainValues))
	diff = maxValue - minValue
	plt.gca().set_ylim(minValue - diff / 10, maxValue + diff / 10)

	# Finally, save the figure with the name of the metric
	plt.savefig("%s.png" % (metric), dpi=dpi)

def getModelHistoryMessage(model):
		Str = model.summary() + "\n"
		trainHistory = model.trainHistory
		for i in range(len(trainHistory)):
			Str += trainHistory[i]["message"] + "\n"
		return Str