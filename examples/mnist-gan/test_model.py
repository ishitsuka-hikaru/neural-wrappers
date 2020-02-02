import torch as tr
import numpy as np
from neural_wrappers.pytorch import device
import matplotlib.pyplot as plt

def test_model(args, model):
	while True:
		randomInputsG = np.random.randn(args.batch_size, args.latent_space_size).astype(np.float32)
		randomOutG = model.generator.npForward(randomInputsG)
		outD = model.discriminator.npForward(randomOutG)
		for j in range(args.batch_size):
			plt.imshow(randomOutG[j, :, :, 0])
			plt.title("Discriminator: %2.3f" % (outD[j]))
			plt.show()