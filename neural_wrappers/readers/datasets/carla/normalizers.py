import numpy as np
from typing import Dict, Union
from .utils import getQuatFromRotation
from ....utilities import resize_batch, h5ReadDict, npGetInfo
from ...dataset_reader import DatasetReader

# TODO: All norms now put data in [0 : 1]. We should look at the rederObj and if some dims want other range, transform
#  the data to that range.

def rgbNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	# x [MBx854x854x3] => [MBx256x256x3] :: [0 : 255]
	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

def depthNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	depthStats = {"min" : 0, "max" : readerObj.hyperParameters["maxDepthMeters"]}

	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# Depth is stored in [0 : 1] representing up to 1000m from simulator
	x = np.clip(x * 1000, depthStats["min"], depthStats["max"])
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	return np.expand_dims(x, axis=-1)

def positionNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation, rotation = x[:, 0 : 3], x[:, 3 :]
	# Now, just for [0 : 1]
	translation = (translation - minPos) / (maxPos - minPos)
	rotation = ((rotation / 180) + 1) / 2
	position = np.concatenate([translation, rotation], axis=-1)

	return position

def positionQuatNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation, rotation = x[:, 0 : 3], x[:, 3 :]
	# Now, just for [0 : 1]
	# Translation is easy, just min max it
	translation = (translation - minPos) / (maxPos - minPos)

	# Rotation is in [-180 : 180], move to [-1 : 1], then call getQuatFromRotation
	rotation = rotation / 180
	quatRotation = getQuatFromRotation(rotation)
	# The returned quaternion is in [-1 : 1], we move it to [0 : 1]
	quatRotation = (quatRotation + 1) / 2
	position = np.concatenate([translation, quatRotation], axis=-1)

	return position

def positionDotTranslationOnlyNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	import cv2
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])
	radius = readerObj.hyperParameters["dotRadius"]
	assert not radius is None

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation = x[:, 0 : 3]
	# Now, just for [0 : 1]
	# Translation is easy, just min max it
	translation = (translation - minPos) / (maxPos - minPos)
	translation = translation[:, 0 : 2]
	MB = translation.shape[0]
	positionDot = np.zeros((MB, readerObj.desiredShape[0], readerObj.desiredShape[1]), dtype=np.float32)

	for i in range(MB):
		translation_x, translation_y = translation[i]
		center_x = int(translation_x * readerObj.desiredShape[1])
		center_y = int(translation_y * readerObj.desiredShape[0])
		positionDot[i] = cv2.circle(positionDot[i], (center_x, center_y), radius=radius, color=1, thickness=-1)

	positionDot = np.expand_dims(positionDot, axis=-1)
	return positionDot

def opticalFlowNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	# Optical flow is in [-1:1] and 100% of percentage. Result is in [0:1] using only [-x%:x%] of data.
	def opticalFlowPercentageTransform(x, opticalFlowPercentage):
		# x :: [0 : 1], centered in 0
		x = (x - 0.5) * 2
		# x :: [-1 : 1], centered in 0
		opticalFlowPercentage = np.array(opticalFlowPercentage) / 100
		flow_x = np.expand_dims(x[..., 0], axis=-1)
		flow_y = np.expand_dims(x[..., 1], axis=-1)
		# flow_x :: [-x% : x%], flow_y :: [-y% : y%]
		flow_x = np.clip(flow_x, -opticalFlowPercentage[0], opticalFlowPercentage[0])
		flow_y = np.clip(flow_y, -opticalFlowPercentage[1], opticalFlowPercentage[1])
		# flow_x in [0 : 2*x%], flow_y :: [0 : 2*y%]
		flow_x += opticalFlowPercentage[0]
		flow_y += opticalFlowPercentage[1]
		# flow_x :: [0 : 1], flow_y :: [0 : 1]
		flow_x *= 1 / (2 * opticalFlowPercentage[0])
		flow_y *= 1 / (2 * opticalFlowPercentage[1])
		# flow :: [0 : 1]
		flow = np.concatenate([flow_x, flow_y], axis=-1).astype(np.float32)
		return flow

	def opticalFlowMagnitude(x):
		# flow :: [0 : 1] => [-1 : 1]
		x = (x - 0.5) * 2
		# norm :: [0 : sqrt(2)] => [0 : 1]
		norm = np.hypot(x[..., 0], x[..., 1]) / np.sqrt(2)
		return np.expand_dims(norm, axis=-1)

	# Data in [0 : 1]
	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")

	if readerObj.hyperParameters["opticalFlowPercentage"] != (100, 100):
		x = opticalFlowPercentageTransform(x, readerObj.hyperParameters["opticalFlowPercentage"])

	if readerObj.hyperParameters["opticalFlowMode"] == "xy":
		return x
	elif readerObj.hyperParameters["opticalFlowMode"] == "magnitude":
		return opticalFlowMagnitude(x)
	elif readerObj.hyperParameters["opticalFlowMode"] == "xy_plus_magnitude":
		return np.concatenate([x, opticalFlowMagnitude(x)], axis=-1)
	assert False

# def opticalFlowNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
# 	# Data in [0 : 1]
# 	width, height = readerObj.desiredShape
# 	x = resize_batch(x, height=height, width=width, resizeLib="opencv")
# 	x[..., 0] = np.float32(np.int32(x[..., 0] * width))
# 	x[..., 1] = np.float32(np.int32(x[..., 1] * height))
# 	return x

def normalNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# Normals are stored as [0 - 255] on 3 channels, representing orientation of the 3 axes.
	x = x.astype(np.float32) / 255
	return x

def semanticSegmentationNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	labelKeys = list({
		(0, 0, 0): "Unlabeled",
		(70, 70, 70): "Building",
		(153, 153, 190): "Fence",
		(160, 170, 250): "Other",
		(60, 20, 220): "Pedestrian",
		(153, 153, 153): "Pole",
		(50, 234, 157): "Road line",
		(128, 64, 128): "Road",
		(232, 35, 244): "Sidewalk",
		(35, 142, 107): "Vegetation",
		(142, 0, 0): "Car",
		(156, 102, 102): "Wall",
		(0, 220, 220): "Traffic sign"
	}.keys())
	numClasses = len(labelKeys)
	labelKeys = list(map(lambda x : x[0] + x[1] * 256 + x[2] * 256 * 256, labelKeys))

	x = x.astype(np.uint32)
	x = x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256
	for i in range(len(labelKeys)):
		x[x == labelKeys[i]] = i
	x = x.astype(np.uint8)
	x = resize_batch(x, interpolation="nearest", height=readerObj.desiredShape[0], \
		width=readerObj.desiredShape[1], resizeLib="opencv")

	# Some fancy way of doing one-hot encoding.
	return np.eye(numClasses)[x].astype(np.float32)