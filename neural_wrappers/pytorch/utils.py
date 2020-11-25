import matplotlib.pyplot as plt
import torch as tr
import numpy as np
import sys
from collections import OrderedDict

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

def trModuleWrapper(module):
	from .feed_forward_network import FeedForwardNetwork
	class Model(FeedForwardNetwork):
		def __init__(self, module):
			super().__init__()
			self.module = module

		def forward(self, x):
			return self.module(x)
	return Model(module)

# Used by NWModule so that we enter a block, we can apply train/eval and when we leave it, we restore the
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

# Results come in torch format, but callbacks require numpy, so convert the results back to numpy format
def npGetData(data):
	if data is None:
		return None
	elif isinstance(data, (int, float)):
		return np.array([data])
	elif isinstance(data, (list, tuple)):
		return [npGetData(x) for x in data]
	elif isinstance(data, (dict, OrderedDict)):
		return {k : npGetData(data[k]) for k in data}
	elif isinstance(data, tr.Tensor):
		return data.detach().to("cpu").numpy()
	elif isinstance(data, np.ndarray):
		return data
	elif callable(data):
		return data
	elif isinstance(data, str):
		return data
	assert False, "Got type %s" % (type(data))

# Equivalent of the function above, but using the data from generator (which comes in numpy format)
def trGetData(data):
	if data is None:
		return None
	elif isinstance(data, (np.int32, np.int8, np.int16, np.int64, np.float32, np.float64, int, float)):
		return tr.Tensor([data]).to(device)
	elif isinstance(data, (list, tuple)):
		return [trGetData(x) for x in data]
	elif isinstance(data, (dict, OrderedDict)):
		return {k : trGetData(data[k]) for k in data}
	elif isinstance(data, tr.Tensor):
		return data.to(device)
	elif isinstance(data, np.ndarray):
		return tr.from_numpy(data).to(device)
	elif callable(data):
		return data
	elif isinstance(data, str):
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

def npToTrCall(fn, *args, **kwargs):
	return npGetData(fn(*trGetData(args), **trGetData(kwargs)))

def trToNpCall(fn, *args, **kwargs):
	return trGetData(fn(*npGetData(args), **npGetData(kwargs)))

def plotModelMetricHistory(trainHistory, metricName, plotBestBullet, dpi=120):
	# Aggregate all the values from trainHistory into a list and plot them
	numEpochs = len(trainHistory)
	# trainValues = np.array([trainHistory[i]["Train"][metric] for i in range(numEpochs)])

	trainValues, validationValues = np.zeros((2, numEpochs), dtype=np.float32)
	hasValidation = ("Validation" in trainHistory[0]) and (not trainHistory[0]["Validation"] is None)
	for i in range(numEpochs):
		X = getMetricScoreFromHistory(trainHistory[i]["Train"], metricName)
		# Apply mean in case the metric returned a tuple.
		trainValues[i] = X.mean()
		
		if hasValidation:
			X = getMetricScoreFromHistory(trainHistory[i]["Validation"], metricName)
			validationValues[i] = X.mean()

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

	assert plotBestBullet in ("none", "min", "max"), "%s" % plotBestBullet
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

# Metrics may be stored as tuples for graph/complex networks.
def getMetricScoreFromHistory(trainHistory, metricName):
	score = trainHistory
	for i in range(len(metricName.name) - 1):
		score = score[metricName.name[i]]
	score = score[metricName]
	return score

def _getOptimizerStr(optimizer):
	if isinstance(optimizer, dict):
		return ["Dict"]

	if optimizer is None:
		return ["None"]

	if isinstance(optimizer, tr.optim.SGD):
		groups = optimizer.param_groups[0]
		params = "Learning rate: %s, Momentum: %s, Dampening: %s, Weight Decay: %s, Nesterov: %s" % \
			(groups["lr"], groups["momentum"], groups["dampening"], groups["weight_decay"], groups["nesterov"])
		optimizerType = "SGD"
	elif isinstance(optimizer, (tr.optim.Adam, tr.optim.AdamW)):
		groups = optimizer.param_groups[0]
		params = "Learning rate: %s, Betas: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], groups["betas"], \
			groups["eps"], groups["weight_decay"])
		optimizerType = {
			tr.optim.Adam : "Adam",
			tr.optim.AdamW : "AdamW"
		}[type(optimizer)]
	elif isinstance(optimizer, tr.optim.RMSprop):
		groups = optimizer.param_groups[0]
		params = "Learning rate: %s, Momentum: %s. Alpha: %s, Eps: %s, Weight Decay: %s" % (groups["lr"], \
			groups["momentum"], groups["alpha"], groups["eps"], groups["weight_decay"])
		optimizerType = "RMSprop"
	elif isinstance(optimizer, tr.optim.Optimizer):
		return str(optimizer)
	else:
		optimizerType = "Generic Optimizer"
		params = str(optimizer)

	return ["%s. %s" % (optimizerType, params)]
