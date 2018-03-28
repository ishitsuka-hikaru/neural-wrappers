from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu
import numpy as np
import torch as tr

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

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
		inputs = maybeCuda(Variable(tr.from_numpy(np.float32(np.random.randn(N, I)))))
		targets = maybeCuda(Variable(tr.from_numpy(np.float32(np.random.randn(N, O)))))

		model = maybeCuda(Model(I, H, O))
		for i in range(5):
			outputs = model.forward(inputs)
			error = tr.sum((outputs - targets)**2)

			error.backward()
			for param in model.parameters():
				param.data -= 0.001 * param.grad.data
				param.grad *= 0

		model.save_weights("test_weights.pkl")
		model_new = maybeCuda(Model(I, H, O))
		model_new.load_weights("test_weights.pkl")

		outputs = model.forward(inputs)
		error = maybeCpu(tr.sum((outputs - targets)**2)).data.numpy()
		outputs_new = model_new.forward(inputs)
		error_new = maybeCpu(tr.sum((outputs_new - targets)**2)).data.numpy()
		assert np.abs(error - error_new) < 1e-5

	def test_save_model_1(self):
		N, I, H, O = 500, 100, 50, 30
		inputs = np.float32(np.random.randn(N, I))	
		targets = np.float32(np.random.randn(N, O))

		model = maybeCuda(Model(I, H, O))
		model.setOptimizer(Adam, lr=0.001)
		model.setCriterion(lambda y, t : tr.sum((y - t)**2))

		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)
		model.save_model("test_model.pkl")
		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)
		model_new = maybeCuda(Model(I, H, O))
		model_new.load_model("test_model.pkl")
		model_new.setCriterion(lambda y, t : tr.sum((y - t)**2))
		model_new.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)

		weights_model = list(model.parameters())
		weights_model_new = list(model_new.parameters())

		assert len(weights_model) == len(weights_model_new)
		for j in range(len(weights_model)):
			weight = weights_model[j]
			weight_new = weights_model_new[j]
			diff = maybeCpu(tr.sum(tr.abs(weight - weight_new))).data.numpy()
			assert diff < 1e-5, "%d: Diff: %2.5f.\n %s %s" % (j, diff, weight, weight_new)

if __name__ == "__main__":
	TestNetwork().test_save_weights_1()
	TestNetwork().test_save_model_1()