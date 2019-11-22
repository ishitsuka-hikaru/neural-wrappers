import numpy as np
from .node import VectorNode
from functools import partial

class Pose(VectorNode):
	def __init__(self, positionsExtremes):
		self.positionsExtremes = positionsExtremes
		metrics = {
			"Position (m)" : partial(Pose.positionMetric, positionsExtremes=positionsExtremes),
			"Orientation (deg)" : partial(Pose.orientationMetric, positionsExtremes=positionsExtremes)
		}
		super().__init__("Pose", numDims=7, lossFn=Pose.lossFn, metrics=metrics, groundTruthKey="pose")

	def lossFn(y, t):
		return ((y - t)**2).mean()

	def positionMetric(y, t, keyName, positionsExtremes, **k):
		Min, Max = positionsExtremes["min"], positionsExtremes["max"]
		# Clip output and renormalie to meters. Assume output and targets are in [0:1]
		tPosition = t["pose"][:, 0 : 3] * (Max-  Min) + Min
		yPosition = np.clip(y[keyName][:, 0 : 3], 0, 1) * (Max - Min) + Min
		return np.abs(yPosition - tPosition).mean()

	def orientationMetric(y, t, keyName, **k): 
		tOrientation = t["pose"][:, 3 : ]
		yOrientation = y[keyName][:, 3 :]

		tOrientation = np.array([txe.quat2euler(tOrientation[i]) for i in range(len(tOrientation))]) + np.pi
		yOrientation = np.array([txe.quat2euler(yOrientation[i]) for i in range(len(tOrientation))]) + np.pi

		Diff = np.abs(tOrientation - yOrientation)
		Diff = Diff / (2 * np.pi) * 360
		return Diff.mean()