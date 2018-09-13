import sys
import numpy as np
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.callbacks import SaveModels
from neural_wrappers.utilities import toCategorical
import matplotlib.pyplot as plt
from scipy.misc import toimage

import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def plot_images(images, titles):
	fig = plt.figure()
	numImages = len(images)

	for j in range(numImages):
		fig.add_subplot(1, numImages, j + 1)
		plt.imshow(np.array(toimage(images[j])), cmap="gray")
		plt.title(titles[j])
		plt.axis("off")
	plt.show()

class BinaryMNISTReader(MNISTReader):
	def iterate_once(self, type, miniBatchSize):
		for items in super().iterate_once(type, miniBatchSize):
			images, targets = items
			# Images are N(0, I), so we can threshold at 0 to get binary values.
			images = np.float32(images > 0)
			targets = np.float32(targets)
			yield [images, targets], images

class FCEncoder(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		# 10 classes (one-hot encoded) in the first layer
		self.fc1 = nn.Linear(28 * 28 + 10, 100)
		self.fc2 = nn.Linear(100, 100)
		self.mean_fc = nn.Linear(100, numEncodings)
		self.mean_std = nn.Linear(100, numEncodings)

	def forward(self, x, t):
		x = x.view(-1, 28 * 28)
		x_concat = tr.cat([x, t], dim=1)
		y1 = F.relu(self.fc1(x_concat))
		y2 = F.relu(self.fc2(y1))
		y_mean = self.mean_fc(y2)
		y_std = self.mean_std(y2)
		return y_mean, y_std

class Decoder(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		self.fc1 = nn.Linear(numEncodings + 10, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, z_samples, t):
		z_concat = tr.cat([z_samples, t], dim=1)
		y1 = F.relu(self.fc1(z_concat))
		y2 = self.fc2(y1)
		y_decoder = tr.sigmoid(y2)
		return y_decoder

class VAE(NeuralNetworkPyTorch):
	def __init__(self, numEncodings, encoderType="FCEncoder"):
		super().__init__()
		assert encoderType in ("FCEncoder", "ConvEncoder")
		self.numEncodings = numEncodings
		if encoderType == "FCEncoder":
			self.encoder = FCEncoder(numEncodings)
		else:
			self.encoder = ConvEncoder(numEncodings)
		self.decoder = Decoder(numEncodings)

	def forward(self, x):
		x, t = x
		batchSize = x.shape[0]
		y_mean, y_std = self.encoder(x, t)
		# "Reparametrization trick": Sample from N(0, I) and multiply by our distribution's mean/std.
		z_samples = maybeCuda(tr.randn(batchSize, self.numEncodings))
		z_samples *= y_std
		z_samples += y_mean
		y_decoder = self.decoder(z_samples, t)
		y_decoder = y_decoder.view(batchSize, 28, 28)
		return y_decoder, y_mean, y_std

def lossFunction(y_network, y_target):
	y_decoder, y_mean, y_std = y_network
	# KL-Divergence between two Gaussians, one with y_mean, y_std and other is N(0, I)
	latent_loss = 0.5 * tr.sum((y_std**2 + y_mean**2 - 1 - tr.log(y_std**2)))
	# decoder_loss = tr.mean((y_decoder - y_target)**2)
	decoder_loss = F.binary_cross_entropy(y_decoder, y_target)
	# ab = (latent_loss / decoder_loss).detach().float()
	return latent_loss + 28 * 28 * 10 * decoder_loss

def lossLatent(y_network, y_target, **kwargs):
	y_decoder, y_mean, y_std = y_network
	latent_loss = 0.5 * np.sum((y_std**2 + y_mean**2 - 1 - np.log(y_std**2)))
	return latent_loss

def lossDecoder(y_network, y_target, **kwargs):
	y_decoder, y_mean, y_std = y_network
	decoder_loss = F.binary_cross_entropy(
		tr.from_numpy(y_decoder).cpu(), \
		tr.from_numpy(y_target).cpu()
	).numpy()
	return decoder_loss

def main():
	miniBatchSize = 100
	numEpochs = 2000

	reader = BinaryMNISTReader(sys.argv[2])
	generator = reader.iterate("train", miniBatchSize=miniBatchSize)
	numIterations = reader.getNumIterations("train", miniBatchSize=miniBatchSize)
	print("Batch size: %d. Num iterations: %d" % (miniBatchSize, numIterations))
	network = maybeCuda(VAE(numEncodings=300))
	network.setCriterion(lossFunction)
	network.setOptimizer(optim.SGD, lr=0.000001, momentum=0.3)
	metrics = { "Latent Loss" : lossLatent, "Decoder Loss" : lossDecoder }
	network.setMetrics(metrics)
	print(network.summary())

	if sys.argv[1] == "train":
		callbacks = [SaveModels("last")]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "retrain":
		callbacks = [SaveModels("last")]
		network.loadModel(sys.argv[3])
		callbacks[0].best = network.trainHistory[-1]["trainMetrics"]["Loss"]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "test":
		network.loadModel(sys.argv[3])
		while True:
			z_samples = maybeCuda(tr.randn(1, network.numEncodings))
			desiredClass = np.random.randint(0, 10, size=(1, 1))
			print(desiredClass)
			t = maybeCuda(tr.from_numpy(np.float32(toCategorical(desiredClass, 10))))
			y_result = network.decoder.forward(z_samples, t)
			result = maybeCpu(y_result.data).numpy().reshape((28, 28))
			result_binary = np.uint8(result > 0.5)
			plot_images([result, result_binary], ["Sampled image", "Binary"])

if __name__ == "__main__":
	main()