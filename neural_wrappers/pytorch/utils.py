import torch as tr
import numpy as np
import sys

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
		return results
	if type(results) in (list, tuple):
		npResults = []
		for result in results:
			npResult = getNpData(result)
			npResults.append(npResult)
	elif type(results) == tr.Tensor:
		 npResults = maybeCpu(results.detach()).numpy()
	else:
		assert False, "Got type %s" % (type(results))
	return npResults

# Equivalent of the function above, but using the data from generator (which comes in numpy format)
def getTrData(data):
	trData = None
	if data is None:
		return data
	elif type(data) in (list, tuple):
		trData = []
		for item in data:
			trItem = getTrData(item)
			trData.append(trItem)
	elif type(data) is np.ndarray:
		trData = maybeCuda(tr.from_numpy(data))
	elif type(data) is tr.Tensor:
		trData = maybeCuda(data)
	return trData