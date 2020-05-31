import matplotlib.pyplot as plt
import torch as tr
import numpy as np
import sys
from collections import OrderedDict

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

def trModuleWrapper(module):
	from .network import NeuralNetworkPyTorch
	class Model(NeuralNetworkPyTorch):
		def __init__(self, module):
			super().__init__()
			self.module = module

		def forward(self, x):
			return self.module(x)
	return Model(module)

# Used by NeuralNetworkPyTorch so that we enter a block, we can apply train/eval and when we leave it, we restore the
#  previous state.
class StorePrevState:
	def __init__(self, moduleObj):
		self.moduleObj = moduleObj

	def __enter__(self):
		self.prevState = self.moduleObj.train if self.moduleObj.training else self.moduleObj.eval

	def __exit__(self, type, value, traceback):
		self.prevState()

def getTrainableParameters(model):
	if not model.training:
		return {}

	trainableParameters = {}
	namedParams = dict(model.named_parameters())
	# Some PyTorch "weird" stuff. Basically this is a hack specifically for BatchNorm (Dropout not supported yet...).
	# BatchNorm parameters are not stored in named_parameters(), just in state_dict(), however in state_dict() we can't
	#  know if it's trainable or not. So, in order to keep all trainable parameters, we need to check if it's either
	#  a BN (we'll also store non-trainable BN, but that's okay) or if it's trainable (in named_params).
	def isBatchNormModuleTrainable(name):
		nonParametersNames = ["running_mean", "running_var", "num_batches_tracked"]
		if name.split(".")[-1] in nonParametersNames:
			# edges.10.model.module.0.conv7.1.running_mean => edges.10.model.module.0.conv7.1.weight is trainable?
			resName = ".".join(name.split(".")[0 : -1])
			potentialName = "%s.weight" % (resName)
			if potentialName in namedParams and namedParams[potentialName].requires_grad:
				return True
		return False

	for name in model.state_dict():
		if isBatchNormModuleTrainable(name):
			trainableParameters[name] = model.state_dict()[name]

		if (name in namedParams) and (namedParams[name].requires_grad):
			trainableParameters[name] = model.state_dict()[name]
	return trainableParameters

def _computeNumParams(namedParams):
	numParams = 0
	for name in namedParams:
		param = namedParams[name]
		numParams += np.prod(param.shape)
	return numParams

def getNumParams(model):
	return _computeNumParams(model.state_dict()), _computeNumParams(getTrainableParameters(model))

def getOptimizerStr(optimizer):
	if optimizer is None:
		return "None"

	groups = optimizer.param_groups[0]
	if type(optimizer) == tr.optim.SGD:
		params = "Learning rate: %s, Momentum: %s, Dampening: %s, Weight Decay: %s, Nesterov: %s" % (groups["lr"], \
			groups["momentum"], groups["dampening"], groups["weight_decay"], groups["nesterov"])
	elif type(optimizer) in (tr.optim.Adam, tr.optim.AdamW):
		params = "Learning rate: %s, Betas: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], groups["betas"], \
			groups["eps"], groups["weight_decay"])
	elif type(optimizer) == tr.optim.RMSprop:
		params = "Learning rate: %s, Momentum: %s. Alpha: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], \
			groups["momentum"], groups["alpha"], groups["eps"], groups["weight_decay"])
	else:
		raise NotImplementedError("Not yet implemneted optimizer str for %s" % (type(optimizer)))

	optimizerType = {
		tr.optim.SGD : "SGD",
		tr.optim.Adam : "Adam",
		tr.optim.AdamW : "AdamW",
		tr.optim.RMSprop : "RMSprop"
	}[type(optimizer)]

	return "%s. %s" % (optimizerType, params)

# Results come in torch format, but callbacks require numpy, so convert the results back to numpy format
def getNpData(data):
	if data is None:
		return None
	elif type(data) in (list, tuple):
		return [getNpData(x) for x in data]
	elif type(data) in (dict, OrderedDict):
		return {k : getNpData(data[k]) for k in data}
	elif type(data) == tr.Tensor:
		return data.detach().to("cpu").numpy()
	elif type(data) == np.ndarray:
		return data
	elif callable(data):
		return data
	assert False, "Got type %s" % (type(data))

# Equivalent of the function above, but using the data from generator (which comes in numpy format)
def getTrData(data):
	if data is None:
		return None
	elif type(data) in (list, tuple):
		return [getTrData(x) for x in data]
	elif type(data) in (dict, OrderedDict):
		return {k : getTrData(data[k]) for k in data}
	elif type(data) == np.ndarray:
		return tr.from_numpy(data).to(device)
	elif type(data) == tr.Tensor:
		return data.to(device)
	elif callable(data):
		return data
	assert False, "Got type %s" % (type(data))

# Equivalent of function above but does detach()
def trDetachData(data):
	if data is None:
		return None
	elif type(data) in (list, tuple):
		return [trDetachData(x) for x in data]
	elif type(data) in (dict, OrderedDict):
		return {k : trDetachData(data[k]) for k in data}
	elif type(data) is tr.Tensor:
		return data.detach()
	assert False, "Got type %s" % (type(data))

def trNpCall(fn, *args, **kwargs):
	return getNpData(fn(*getTrData(args), **getTrData(kwargs)))

def npTrCall(fn, *args, **kwargs):
	return getTrData(fn(*getNpData(args), **getNpData(kwargs)))

def plotModelMetricHistory(trainHistory, metricName, plotBestBullet, dpi=120):
	# Aggregate all the values from trainHistory into a list and plot them
	numEpochs = len(trainHistory)
	# trainValues = np.array([trainHistory[i]["Train"][metric] for i in range(numEpochs)])

	def getScore(trainHistory, name):
		score = trainHistory
		for k in name:
			score = score[k]
		return score

	trainValues, validationValues = np.zeros((2, numEpochs), dtype=np.float32)
	hasValidation = ("Validation" in trainHistory[0]) and (not trainHistory[0]["Validation"] is None)
	for i in range(numEpochs):
		trainValues[i] = getScore(trainHistory[i]["Train"], metricName)
		if hasValidation:
			validationValues[i] = getScore(trainHistory[i]["Validation"], metricName)

	x = np.arange(len(trainValues)) + 1
	metricName = str(metricName)
	plt.gcf().clf()
	plt.gca().cla()
	plt.plot(x, trainValues, label="Train %s" % (metricName))

	if hasValidation:
		plt.plot(x, validationValues, label="Val %s" % (metricName))
		usedValues = np.array(validationValues)
	else:
		usedValues = trainValues
	# Against NaNs killing the training for low data count.
	trainValues[np.isnan(trainValues)] = 0
	usedValues[np.isnan(usedValues)] = 0

	assert plotBestBullet in ("none", "min", "max")
	if plotBestBullet == "min":
		minX, minValue = np.argmin(usedValues), np.min(usedValues)
		plt.annotate("Epoch %d\nMin %2.2f" % (minX + 1, minValue), xy=(minX + 1, minValue))
		plt.plot([minX + 1], [minValue], "o")
	elif plotBestBullet == "max":
		maxX, maxValue = np.argmax(usedValues), np.max(usedValues)
		plt.annotate("Epoch %d\nMax %2.2f" % (maxX + 1, maxValue), xy=(maxX + 1, maxValue))
		plt.plot([maxX + 1], [maxValue], "o")

	# Set the y axis to have some space above and below the plot min/max values so it looks prettier.
	minValue = min(np.min(usedValues), np.min(trainValues))
	maxValue = max(np.max(usedValues), np.max(trainValues))
	diff = maxValue - minValue
	plt.gca().set_ylim(minValue - diff / 10, maxValue + diff / 10)

	# Finally, save the figure with the name of the metric
	plt.xlabel("Epoch")
	plt.ylabel(metricName)
	plt.legend()
	plt.savefig("%s.png" % (metricName), dpi=dpi)

def getModelHistoryMessage(model):
		Str = model.summary() + "\n"
		trainHistory = model.trainHistory
		for i in range(len(trainHistory)):
			Str += trainHistory[i]["message"] + "\n"
		return Str