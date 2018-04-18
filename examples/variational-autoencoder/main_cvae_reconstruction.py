# Implementation of Conditional Variational Auto Encoder (CVAE).
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

from main import FCEncoder, ConvEncoder, VAE, lossFunction, lossLatent, lossDecoder
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.callbacks import SaveModels

import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class ConditionalBinaryMNIST(MNISTReader):
	def iterate_once(self, type, miniBatchSize):
		for items in super().iterate_once(type, miniBatchSize):
			images, _ = items
			# Images are N(0, I), so we can threshold at 0 to get binary values.
			images = images > 0
			# corruptedImages = corruptImages_pepper(images, threshold=0.7)
			corruptedImages = corruptImages_erase(images)
			givenImages = np.zeros((2, *images.shape), dtype=np.float32)
			givenImages[0] = np.float32(images)
			givenImages[1] = np.float32(corruptedImages)
			yield givenImages, np.float32(images)

class Decoder(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		self.fc1 = nn.Linear(numEncodings + 28 * 28, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, x_corrupted, z_samples):
		x_corrupted = x_corrupted.view(-1, 28 * 28)
		x_cat = tr.cat([x_corrupted, z_samples], dim=1)
		y1 = F.relu(self.fc1(x_cat))
		y2 = self.fc2(y1)
		y_decoder = F.sigmoid(y2)
		return y_decoder

class CVAE(VAE):
	def __init__(self, numEncodings, encoderType="FCEncoder"):
		super().__init__(numEncodings, encoderType)
		self.decoder = Decoder(numEncodings)

	def forward(self, x):
		x_corrupted, x_original = x
		batchSize = x_corrupted.shape[0]
		y_mean, y_std = self.encoder(x_original)
		# "Reparametrization trick": Sample from N(0, I) and multiply by our distribution's mean/std.
		z_samples = Variable(maybeCuda(tr.randn(batchSize, self.numEncodings)), requires_grad=False)
		z_samples *= y_std
		z_samples += y_mean
		y_decoder = self.decoder(x_corrupted, z_samples)
		y_decoder = y_decoder.view(batchSize, 28, 28)
		return y_decoder, y_mean, y_std

def corruptImages_pepper(images, threshold=0.95):
	# Corrupt 5% of the pixels
	whereCorrupt = np.uint8(np.random.rand(*images.shape) > threshold)
	images = np.uint8(np.abs(np.int32(images) - whereCorrupt))
	return images

def corruptImages_erase(images):
	newImages = np.copy(images)
	for j in range(len(images)):
		boxShape = np.random.randint(4, 8, size=(2, ))
		startPositions_i = np.random.randint(5, 28 - 5 - boxShape[0] + 1)
		startPositions_j = np.random.randint(5, 28 - 5 - boxShape[1] + 1)
		newImages[j, startPositions_i : startPositions_i + boxShape[0], \
			startPositions_j : startPositions_j + boxShape[1]] = 0
	return newImages

def plot_images(images, titles):
	fig = plt.figure()
	numImages = len(images)

	for j in range(numImages):
		fig.add_subplot(1, numImages, j + 1)
		plt.imshow(np.array(toimage(images[j])), cmap="gray")
		plt.title(titles[j])
		plt.axis("off")
	plt.show()

def main():
	assert len(sys.argv) >= 4, "Usage: python main.py <train/test/retrain/test_autoencoder> <path/to/mnist.h5> " + \
		"<FCEncoder/ConvEncoder> [model]"
	miniBatchSize = 100
	numEpochs = 2000

	reader = ConditionalBinaryMNIST(sys.argv[2])
	generator = reader.iterate("train", miniBatchSize=miniBatchSize)
	numIterations = reader.getNumIterations("train", miniBatchSize=miniBatchSize)

	print("Batch size: %d. Num iterations: %d" % (miniBatchSize, numIterations))
	network = maybeCuda(CVAE(numEncodings=300, encoderType=sys.argv[3]))
	network.setCriterion(lossFunction)
	network.setOptimizer(optim.SGD, lr=0.000001, momentum=0.3)
	metrics = { "Loss" : lambda x, y, **k : k["loss"], "Latent Loss" : lossLatent, "Decoder Loss" : lossDecoder }
	network.setMetrics(metrics)
	print(network.summary())

	if sys.argv[1] == "train":
		callbacks = [SaveModels("best")]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "retrain":
		assert len(sys.argv) == 5
		callbacks = [SaveModels("best")]
		network.load_model(sys.argv[4])
		callbacks[0].best = network.trainHistory[-1]["trainMetrics"]["Loss"]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "test":
		assert len(sys.argv) == 5
		network.load_model(sys.argv[4])

		generator = reader.iterate("test", miniBatchSize=miniBatchSize)
		numIterations = reader.getNumIterations("test", miniBatchSize=miniBatchSize)

		for items in generator:
			(x_original, x_corrupted), _ = items
			z_samples = Variable(maybeCuda(tr.randn(miniBatchSize, network.numEncodings)), requires_grad=False)
			x_corrupted_tr = Variable(maybeCuda(tr.from_numpy(x_corrupted)), requires_grad=False)

			y_results = network.decoder.forward(x_corrupted_tr, z_samples)
			for j in range(len(y_results)):
				y_result = y_results[j]
				result = maybeCpu(y_result.data).numpy().reshape((28, 28))
				result_binary = np.uint8(result > 0.5)
				plot_images([x_original[j], x_corrupted[j], result, result_binary], \
					["Original", "Corrupted", "Reconstruction", "Binary"])

if __name__ == "__main__":
	main()