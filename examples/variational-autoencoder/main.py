# Implementation of Variational Auto Encoder (VAE) based on http://kvfrans.com/variational-autoencoders-explained/
#  and https://arxiv.org/abs/1606.05908 using binary MNIST.
# Notes: loss is used using the empirical MB * 28 * 28 * reconstruction_loss + latent_loss
# If you get NaNs during training, lower the learning rate or change the optimizer.
import sys
import numpy as np
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.callbacks import SaveModels
import matplotlib.pyplot as plt
from scipy.misc import toimage

import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class BinaryMNISTReader(MNISTReader):
	def iterate_once(self, type, miniBatchSize):
		for items in super().iterate_once(type, miniBatchSize):
			images, _ = items
			# Images are N(0, I), so we can threshold at 0 to get binary values.
			images = images > 0
			yield np.float32(images), np.float32(images)

class Encoder(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		# self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1)
		# self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1)
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

class Decoder(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		self.fc1 = nn.Linear(numEncodings, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, z_samples):
		y1 = F.relu(self.fc1(z_samples))
		y2 = self.fc2(y1)
		y_decoder = F.sigmoid(y2)
		return y_decoder

class VAE(NeuralNetworkPyTorch):
	def __init__(self, numEncodings):
		super().__init__()
		self.numEncodings = numEncodings
		self.encoder = Encoder(numEncodings)
		self.decoder = Decoder(numEncodings)

	def forward(self, x):
		batchSize = x.shape[0]
		y_mean, y_std = self.encoder(x)
		# "Reparametrization trick": Sample from N(0, I) and multiply by our distribution's mean/std.
		z_samples = Variable(maybeCuda(tr.randn(batchSize, self.numEncodings)), requires_grad=False)
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
	return latent_loss + 76800 * decoder_loss

def lossLatent(y_network, y_target, **kwargs):
	y_decoder, y_mean, y_std = y_network
	latent_loss = 0.5 * np.sum((y_std**2 + y_mean**2 - 1 - np.log(y_std**2)))
	return latent_loss

def lossDecoder(y_network, y_target, **kwargs):
	y_decoder, y_mean, y_std = y_network
	decoder_loss = (F.binary_cross_entropy(Variable(tr.from_numpy(y_decoder).cpu()), \
		Variable(tr.from_numpy(y_target)).cpu())).data.numpy()
	return 76800 * decoder_loss

def plot_images(image1, image2, title1="", title2=""):
	fig = plt.figure()
	fig.add_subplot(1, 2, 1)
	plt.imshow(np.array(toimage(image1)), cmap="gray")
	plt.title(title1)
	plt.axis("off")
	fig.add_subplot(1, 2, 2)
	plt.imshow(np.array(toimage(image2)), cmap="gray")
	plt.title(title2)
	plt.axis("off")
	plt.show()

def main():
	assert len(sys.argv) >= 3, "Usage: python main.py <train/test/retrain/test_autoencoder> <path/to/mnist.h5> [model]"
	miniBatchSize = 100
	numEpochs = 2000

	reader = BinaryMNISTReader(sys.argv[2])
	generator = reader.iterate("train", miniBatchSize=miniBatchSize)
	numIterations = reader.getNumIterations("train", miniBatchSize=miniBatchSize)
	print("Batch size: %d. Num iterations: %d" % (miniBatchSize, numIterations))
	network = maybeCuda(VAE(numEncodings=300))
	network.setCriterion(lossFunction)
	network.setOptimizer(optim.SGD, lr=0.000001, momentum=0.3)
	metrics = { "Loss" : lambda x, y, **k : k["loss"], "Latent Loss" : lossLatent, "Decoder Loss" : lossDecoder }
	network.setMetrics(metrics)
	print(network.summary())

	if sys.argv[1] == "train":
		callbacks = [SaveModels("best")]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "retrain":
		assert len(sys.argv) == 4
		callbacks = [SaveModels("best")]
		network.load_model(sys.argv[3])
		callbacks[0].best = network.trainHistory[-1]["trainMetrics"]["Loss"]
		network.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, callbacks=callbacks)

	elif sys.argv[1] == "test_autoencoder":
		assert len(sys.argv) == 4
		generator = reader.iterate("test", miniBatchSize=miniBatchSize)
		numIterations = reader.getNumIterations("test", miniBatchSize=miniBatchSize)
		network.load_model(sys.argv[3])

		for items in generator:
			images, _ = items
			y_result, y_mean, y_std = network.forward(Variable(maybeCuda(tr.from_numpy(images))))
			results = maybeCpu(y_result.data).numpy().reshape((-1, 28, 28))
			for j in range(len(results)):
				loss = lossDecoder([results[j].reshape((1, 28, 28)), None, None], images[j].reshape((1, 28, 28)))
				print("Reconstruction loss: %d" % (loss))
				plot_images(images[j], results[j], "Original", "Reconstruction")
				plt.show()

	elif sys.argv[1] == "test":
		assert len(sys.argv) == 4
		network.load_model(sys.argv[3])
		while True:
			z_samples = Variable(maybeCuda(tr.randn(1, network.numEncodings)), requires_grad=False)
			y_result = network.decoder.forward(z_samples)
			result = maybeCpu(y_result.data).numpy().reshape((28, 28))
			result_binary = np.uint8(result > 0.5)
			plot_images(result, result_binary, "Sampled image", "Binary")

if __name__ == "__main__":
	main()