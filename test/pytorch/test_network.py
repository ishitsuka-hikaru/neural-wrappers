from neural_wrappers.pytorch import NeuralNetworkPyTorch
import numpy as np
import torch as tr

import torch.nn as nn
from torch.autograd import Variable 

class Model(NeuralNetworkPyTorch):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__()
		self.fc1 = nn.Linear(inputSize, hiddenSize)
		self.fc2 = nn.Linear(hiddenSize, hiddenSize)
		self.fc3 = nn.Linear(hiddenSize, outputSize)

	def forward(self, x):
		y1 = self.fc1(x)
		y2 = self.fc2(y1)
		y3 = self.fc3(y2)
		return y3

class TestNetwork:
	def test_save_weights_1(self):
		N, I, H, O = 500, 100, 50, 30
		inputs = Variable(tr.from_numpy(np.float32(np.random.randn(N, I))))	
		targets = Variable(tr.from_numpy(np.float32(np.random.randn(N, O))))

		model = Model(I, H, O)
		for i in range(5):
			outputs = model.forward(inputs)
			error = tr.sum((outputs - targets)**2)

			error.backward()
			for param in model.parameters():
				param.data -= 0.001 * param.grad.data
				param.grad *= 0

		model.save_weights("test_weights.pkl")
		model_new = Model(I, H, O)
		model_new.load_weights("test_weights.pkl")

		outputs = model.forward(inputs)
		error = tr.sum((outputs - targets)**2).data.numpy()
		outputs_new = model_new.forward(inputs)
		error_new = tr.sum((outputs_new - targets)**2).data.numpy()
		assert np.abs(error - error_new) < 1e-5

if __name__ == "__main__":
	TestNetwork().test_save_weights_1()