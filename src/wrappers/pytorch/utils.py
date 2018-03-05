import torch as tr
import numpy as np

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

# Get all the parameters of the optimizer state, except the 'params' key, which is stored separated
def getOptimizerHyperParams(optimizer):
	assert len(optimizer.param_groups) == 1
	paramGroups = optimizer.param_groups[0]
	optimizerState = {}
	for key in paramGroups:
		if key == "params":
			continue

		optimizerState[key] = maybeCpu(paramGroups[key])
	return optimizerState

def getOptimizerParamsState(optimizer):
	states = []
	# Just iterate through values, because keys are the weights themselves, and we already save those.
	for param_state in list(optimizer.state.values()):
		# optimizer.state :: [param -> param_state]
		# param_state :: {Str -> state_tensor}
		saved_state = {}
		for key in param_state:
			saved_state[key] = maybeCpu(param_state[key])
		states.append(saved_state)
	return states

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
