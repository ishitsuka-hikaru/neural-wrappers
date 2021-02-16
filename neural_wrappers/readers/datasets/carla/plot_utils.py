import numpy as np
import cv2
import flow_vis
from matplotlib.cm import plasma
from neural_wrappers.utilities import npGetInfo, minMaxPercentile
from media_processing_lib.image import toImage

def semanticToImage(x):
	x = np.argmax(x, axis=-1)
	# possibleColors = [(0, 0, 0), (80, 80, 80), (190, 153, 153), (250, 170, 160), (220, 20, 60), (153, 153, 153), \
	# 	(157, 234, 50), (128, 128, 128), (200, 200, 140), (107, 142, 35), (0, 0, 142), (102, 102, 156), (220, 220, 0)]
	possibleColors = [(0, 0, 0), (70, 70, 70), (153, 153, 190), (160, 170, 250), (60, 20, 220), (153, 153, 153), \
		(50, 234, 157), (128, 64, 128), (232, 35, 244), (35, 142, 107), (142, 0, 0), (156, 102, 102), (0, 220, 220)]
	possibleColors = list(map(lambda x : [x[2], x[1], x[0]], possibleColors))
	newImage = np.zeros((*x.shape, 3), dtype=np.uint8)
	for i in range(len(possibleColors)):
		newImage[x == i] = possibleColors[i]
	return newImage

def depthToImage(x):
	a = np.clip(x, 0, 1)[..., 0]
	b = plasma(a)[..., 0 : 3]
	c = toImage(b)
	return c

def default(x):
	x = np.clip(x, 0, 1)
	x = toImage(x)
	return x

def cameraNormalToImage(x):
	x = np.abs(x)
	x = np.clip(x, 0, 1)
	# TODO: Fix this at data level!
	x[..., 2] = 1 - x[..., 2]
	x = toImage(x)
	return x

def normalToImage(x):
	x = np.abs(x)
	x = np.clip(x, 0, 1)
	x = toImage(x)
	return x

def opticalFlowToImage(x):
	if x.shape[-1] > 1:
		x = minMaxPercentile(x[..., 0 : 2], 5, 95)
		x = (x - 0.5) * 2
		x = np.array(flow_vis.flow_to_color(x))
	x = toImage(x)
	return x

toImageFuncs = {
	"RGB" : default,
	"Depth" : depthToImage,
	# Pose : poseToImage,
	"Semantic" : semanticToImage,
	"Normal" : normalToImage,
	"CameraNormal" : cameraNormalToImage,
	"WireframeRegression" : default,
	"Halftone" : default,
	"OpticalFlow": opticalFlowToImage,
	"Edges": default,
	"SemanticGB": default
}