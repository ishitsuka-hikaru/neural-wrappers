import matplotlib.pyplot as plt
import numpy as np
import os

from neural_wrappers.callbacks import Callback

class PlotCallback(Callback):
	def __init__(self, latentSpaceSize):
		super().__init__("PlotCallback")
		self.latentSpaceSize = latentSpaceSize

		if not os.path.exists("images"):
			os.mkdir("images")

	def onEpochEnd(self, **kwargs):
		GAN = kwargs["model"]

		# Generate 20 random gaussian inputs
		randomNoise = np.random.randn(20, self.latentSpaceSize).astype(np.float32)
		npRandomOutG = GAN.generator.npForward(randomNoise)[..., 0] / 2 + 0.5

		# Plot the inputs and discriminator's confidence in them
		items = [npRandomOutG[j] for j in range(len(npRandomOutG))]
		
		ax = plt.subplots(4, 5)[1]
		for i in range(4):
			for j in range(5):
				ax[i, j].imshow(npRandomOutG[i * 4 + j], cmap="gray")
		plt.savefig("results_epoch%d.png" % (GAN.currentEpoch))