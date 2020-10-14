import torch.nn as nn
import torch.nn.functional as F
from neural_wrappers.pytorch import FeedForwardNetwork

class MyRNN(nn.Module):
	def __init__(self, inputShape, hiddenShape):
		super(MyRNN, self).__init__()
		self.hiddenShape = hiddenShape
		self.Wih = nn.Linear(in_features=inputShape, out_features=hiddenShape)
		self.Whh = nn.Linear(in_features=hiddenShape, out_features=hiddenShape)

	def forward(self, x, h=None):
		assert len(x.shape) == 3
		T, MB = x.shape[0], x.shape[1]
		if h == None:
			h = tr.zeros(MB, self.hiddenShape, requires_grad=False).to(device)
		res = tr.zeros(T, MB, self.hiddenShape, requires_grad=False).to(device)
		for i in range(T):
			a = self.Wih.forward(x[i])
			b = self.Whh.forward(h)
			h = tr.tanh(a + b)
			res[i] = h
		return res, h.unsqueeze(0)

class Model(FeedForwardNetwork):
	def __init__(self, cellType, inputSize, hiddenSize):
		super(Model, self).__init__()
		assert cellType in ("RNN", "GRU", "LSTM", "MyRNN")
		self.hiddenSize = hiddenSize
		self.inputSize = inputSize
		self.outputSize = inputSize
		self.cellType = cellType

		self.rnn = {
			"MyRNN" : MyRNN(inputShape=self.inputSize, hiddenShape=self.hiddenSize),
			"RNN" : nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1),
			"GRU" : nn.GRU(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1),
			"LSTM" : nn.LSTM(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		}[self.cellType]
		self.fc1 = nn.Linear(self.hiddenSize, self.outputSize)

	def forward(self, x):
		input, hidden = x
		input = input.permute(1, 0, 2)
		# print(input.shape)
		output, hidden = self.rnn(input, hidden)
		output = self.fc1(output)
		output = F.softmax(output, dim=-1)
		output = output.permute(1, 0, 2)
		return output, hidden