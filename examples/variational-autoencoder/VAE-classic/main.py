# Implementation of Variational Auto Encoder (VAE) based on http://kvfrans.com/variational-autoencoders-explained/
#  and https://arxiv.org/abs/1606.05908 using binary MNIST.
# Notes: loss is used using the empirical (latent_loss / reconstruction_loss) * reconstruction_loss + latent_loss
# If you get NaNs during training, lower the learning rate or change the optimizer.
import sys
import numpy as np
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import FeedForwardNetwork, device
from neural_wrappers.callbacks import SaveModels
import matplotlib.pyplot as plt
from scipy.misc import toimage

import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BinaryMNISTReader(MNISTReader):
	def iterate_once(self, type, miniBatchSize):
		for items in super().iterate_once(type, miniBatchSize):
			images, _ = items
			# Images are N(0, I), so we can threshold at 0 to get binary values.
			images = np.float32(images > 0)
			yield images, images

class FCEncoder(FeedForwardNetwork):
	def __init__(self, numEncodings):
		super().__init__()
		self.fc1 = nn.Linear(28 * 28, 100)
		self.fc2 = nn.Linear(100, 100)
		self.mean_fc = nn.Linear(100, numEncodings)
		self.mean_std = nn.Linear(100, numEncodings)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y_mean = self.mean_fc(y2)
		y_std = self.mean_std(y2)
		return y_mean, y_std

class ConvEncoder(FeedForwardNetwork):
	def __init__(self, numEncodings):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1)
		self.fc = nn.Linear(24 * 24 * 100, 100)
		self.mean_fc = nn.Linear(100, numEncodings)
		self.mean_std = nn.Linear(100, numEncodings)

	def forward(self, x):
		x = x.view(-1, 1, 28, 28)
		y1 = F.relu(self.conv1(x))
		y2 = F.relu(self.conv2(y1))
		y2 = y2.view(-1, 24 * 24 * 100)
		y3 = F.relu(self.fc(y2))
		y_mean = self.mean_fc(y3)
		y_std = self.mean_std(y3)
		return y_mean, y_std

class Decoder(FeedForwardNetwork):
	def __init__(self, numEncodings):
		super().__init__()
		self.fc1 = nn.Linear(numEncodings, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, z_samples):
		y1 = F.relu(self.fc1(z_samples))
		y2 = self.fc2(y1)
		y_decoder = tr.sigmoid(y2)
		return y_decoder

class VAE(FeedForwardNetwork):
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
		batchSize = x[0].shape[0]
		y_mean, y_std = self.encoder(x)
		# "Reparametrization trick": Sample from N(0, I) and multiply by our distribution's mean/std.
		z_samples = tr.randn(batchSize, self.numEncodings).to(device)
		z_samples *= y_std
		z_samples += y_mean
		y_decoder = self.decoder(z_samples)
		y_decoder = y_decoder.view(batchSize, 28, 28)
		return y_decoder, y_mean, y_std

def lossFunction(y_network, y_target):
	y_decoder, y_mean, y_std = y_network
	# KL-Divergence between two Gaussians, one with y_mean, y_std and other is N(0, I)
	latent_loss = 0.5 * tr.sum((y_std**2 + y_mean**2 - 1 - tr.log(y_std**2)))
	# decoder_loss = tr.mean((y_decoder - y_target)**2)
	decoder_loss = F.binary_cross_entropy(y_decoder, y_target)
	# ab = (latent_loss / decoder_loss).detach().float()
	return latent_loss + 28 * 28 * 100 * decoder_loss

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

	reader = BinaryMNISTReader(sys.argv[2])
	generator = reader.iterate("train", miniBatchSize=miniBatchSize)
	numIterations = reader.getNumIterations("train", miniBatchSize=miniBatchSize)
	print("Batch size: %d. Num iterations: %d" % (miniBatchSize, numIterations))
	network = VAE(numEncodings=300, encoderType=sys.argv[3]).to(device)
	network.setCriterion(lossFunction)
	network.setOptimizer(optim.SGD, lr=0.000001, momentum=0.3)
	metrics = { "Latent Loss" : lossLatent, "Decoder Loss" : lossDecoder }
	network.setMetrics(metrics)
	print(network.summary())

	if sys.argv[1] == "train":
		callbacks = [SaveModels("best")]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "retrain":
		assert len(sys.argv) == 5
		callbacks = [SaveModels("best")]
		network.loadModel(sys.argv[4])
		callbacks[0].best = network.trainHistory[-1]["trainMetrics"]["Loss"]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "test_autoencoder":
		assert len(sys.argv) == 5
		generator = reader.iterate("test", miniBatchSize=miniBatchSize)
		numIterations = reader.getNumIterations("test", miniBatchSize=miniBatchSize)
		network.loadModel(sys.argv[4])

		for items in generator:
			images, _ = items
			y_result, y_mean, y_std = network.npForward(images)
			results = y_result.reshape((-1, 28, 28))
			for j in range(len(results)):
				loss = lossDecoder([results[j].reshape((1, 28, 28)), None, None], images[j].reshape((1, 28, 28)))
				print("Reconstruction loss: %d" % (loss))
				plot_images([images[j], results[j]], ["Original", "Reconstruction"])
				plt.show()

	elif sys.argv[1] == "test":
		assert len(sys.argv) == 5
		network.loadModel(sys.argv[4])
		while True:
			z_samples = tr.randn(1, network.numEncodings).to(device)
			result = network.decoder.npForward(z_samples).reshape((28, 28))
			result_binary = np.uint8(result > 0.5)
			plot_images([result, result_binary], ["Sampled image", "Binary"])

if __name__ == "__main__":
	main()