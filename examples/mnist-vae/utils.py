import matplotlib.pyplot as plt
import numpy as np
from neural_wrappers.callbacks import Callback

class SampleResultsCallback(Callback):
	def onEpochEnd(self, **kwargs):
		model = kwargs["model"]
		noise = np.random.randn(25, model.decoder.noiseSize).astype(np.float32)
		results = model.decoder.npForward(noise)
		results = results.reshape(5, 5, 28, 28) > 0.5

		plt.gcf().clf()
		ax = plt.subplots(5, 5)[1]
		for i in range(5):
			for j in range(5):
				ax[i, j].imshow(results[i, j], cmap="gray")
		plt.savefig("results_%d.png" % (kwargs["epoch"]))
		plt.close()