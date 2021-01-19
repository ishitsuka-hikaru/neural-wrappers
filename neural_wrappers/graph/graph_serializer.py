from ..pytorch.network_serializer import NetworkSerializer, getTrainableParameters, \
	_computeNumParams
from overrides import overrides

class GraphSerializer(NetworkSerializer):
	# Handles loading weights from a model.
	def doLoadWeightsOld2(self, namedTrainableParams, namedLoadedParams, trainableParams, loadedParams):
		print("Loading in old way, where we hope/assume that the order of the keys is identical to the loaded one")
		# Combines names of trainable params with weights from loaded (should apply to all cases) if the loop down
		#  works (Potential bug: RARE CASE WHERE DICT ORDER IS DIFFERENT BUT SAME # OF PARAMS)
		newParams = {}
		for i in range(len(namedTrainableParams)):
			nameTrainableParam = namedTrainableParams[i]
			nameLoadedParam = namedLoadedParams[i]
			trainableParam = trainableParams[nameTrainableParam]
			loadedParam = loadedParams[nameLoadedParam]
			assert trainableParam.shape == loadedParam.shape, "This: %s:%s. Loaded:%s" % \
				(nameLoadedParam, str(trainableParam.shape), str(loadedParam.shape))
			newParams[nameTrainableParam] = loadedParam
		return newParams

	# Handles loading weights from a model.
	# @overrides
	# def doLoadWeights(self, loadedParams):
	# 	trainableParams = getTrainableParameters(self.model)
	# 	numTrainableParams = _computeNumParams(trainableParams)
	# 	numLoadedParams = _computeNumParams(loadedParams)
	# 	assert numLoadedParams == numTrainableParams, "Inconsistent parameters: Loaded: %d vs Model (trainable): %d." \
	# 		% (numLoadedParams, numTrainableParams)

	# 	namedTrainableParams = sorted(list(trainableParams.keys()))
	# 	namedLoadedParams = sorted(list(loadedParams.keys()))
	# 	# Loading in natural way, because keys are identical
	# 	if namedTrainableParams == namedLoadedParams:
	# 		for key in namedTrainableParams:
	# 			assert trainableParams[key].shape == loadedParams[key].shape, "This: %s:%s. Loaded:%s" % \
	# 				(nameLoadedParam, str(trainableParam.shape), str(loadedParam.shape))
	# 		newParams = loadedParams
	# 	else:
	# 		newParams = self.doLoadWeightsOld2(namedTrainableParams, namedLoadedParams, trainableParams, loadedParams)

	# 	# TODO: Make strict=True and add fake params in the if above (including BN/Dropout).
	# 	missing, unexpected = self.model.load_state_dict(newParams, strict=False)
	# 	if len(missing) > 0:
	# 		print("Loaded partial model. Missing %d keys (got %d keys)" % (len(missing), len(newParams)))
	# 	if len(unexpected):
	# 		print("Unexpected %d keys in the loaded model" % (len(unexpected)))
	# 	print("Succesfully loaded weights (%d parameters) " % (numLoadedParams))