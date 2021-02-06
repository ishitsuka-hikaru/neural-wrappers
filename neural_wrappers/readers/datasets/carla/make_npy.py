import numpy as np

# Function that takes a prediction and converts it back to renormalized npy
# Usage: Can be used for pseuodolabels such that the reader normalizes it correctly
def makeNpy(blended, obj):
	Type = type(obj)
	strType = str(Type).split(".")[-1][0:-2]
	assert strType in ("Semantic", "CameraNormal", "Normal", "Wireframe", "Pose", "Depth", \
		"Halftone", "WireframeRegression")
	if strType == "Semantic":
		# def semanticRenorm(blended):
		# 	Keys = [(0, 0, 0), (70, 70, 70), (153, 153, 190), (160, 170, 250), (60, 20, 220), \
		# 		(153, 153, 153), (50, 234, 157), (128, 64, 128), (232, 35, 244), (35, 142, 107), \
		# 		(142, 0, 0), (156, 102, 102), (0, 220, 220)]
		# 	argmaxBlended = np.argmax(blended, axis=-1)
		# 	result = np.zeros((*argmaxBlended.shape, 3), dtype=np.uint8)
		# 	for i in range(13):
		# 		whereI = np.where(argmaxBlended == i)
		# 		result[whereI] = Keys[i]
		# return result

		# We store the results as predictions (soft labels!)
		def semanticRenorm(blended):
			return blended

		result = semanticRenorm(blended)

	elif strType in ("CameraNormal", "Normal", "Halftone", "WireframeRegression"):
		result = np.clip(blended, 0, 1) * 255
	elif strType == "Pose":
		def positionRenorm(x, positionsExtremes):
			Min, Max = positionsExtremes["min"], positionsExtremes["max"]
			y = np.clip(x[0 : 3], 0, 1)
			y = y * (Max - Min) + Min
			return y

		def orientationRenorm(x):
			# x :: [0 : 1] -> [-1 : 1]
			y = np.clip(x[3 :], 0, 1) * 2 - 1
			# y :: [-1 : 1] -> [-pi : pi]
			# y = np.array(txe.quat2euler(y, "sxyx"))
			# y :: [-pi : pi] -> [-1 : 1]
			# y /= np.pi
			# y :: [-1 : 1] -> [-180 : 180]
			y *= 180
			return y
		translation = positionRenorm(blended, obj.positionsExtremes)
		orientation = orientationRenorm(blended)
		result = np.concatenate([translation, orientation])
	elif strType == "Depth":
		# result :: [0 : 1] representing [0 : maxDepthMeters]. We need to renormalize it to [0 : 1000m]
		result = np.clip(blended, 0, 1)[..., 0]
		# result :: [0 : 1] => [0 : maxDepthMeters]
		result = result * obj.maxDepthMeters
		# result :: [0 : maxDepthMeters] => [0 : 1] where 1 represents 1000m
		result = result / 1000

	return result