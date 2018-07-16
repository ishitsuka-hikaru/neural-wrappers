import sys
import numpy as np
import torch.nn as nn
import torch as tr
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from neural_wrappers.pytorch import RecurrentNeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.callbacks import SaveModels, Callback
from neural_wrappers.models.resnet50_notop import ResNet50NoTop
from neural_wrappers.utilities import minMaxNormalizeData
from scipy.misc import toimage
from Mihlib import readVideo, saveVideo, plot_image, show_plots, npGetInfo
from functools import partial
import moviepy.editor as mpy
import pims

hiddenState = None

class RecurrentCNN(RecurrentNeuralNetworkPyTorch):
	def __init__(self, inputSize=(240, 320, 3), hiddenSize=300, outputSize=(30, 50, 3)):
		super().__init__()

		self.baseModel = ResNet50NoTop()
		for param in self.baseModel.parameters():
			param.requires_grad_(False)
		self.conv1 = nn.Conv2d(in_channels=2048, out_channels=50, kernel_size=1)

		# LSTM part
		self.lstm1 = nn.RNN(input_size=(50*8*10), hidden_size=100, num_layers=1)
		self.fc1 = nn.Linear(in_features=100, out_features=100)
		self.fc2 = nn.Linear(in_features=100, out_features=(30*50*3))

	def forward(self, input, hidden):
		input = tr.transpose(tr.transpose(input, 1, 3), 2, 3) # Move depth first

		y1 = self.baseModel(input)
		y2 = F.relu(self.conv1(y1))
		y2 = y2.view(1, -1, int(np.prod(y2.shape[1 : ])))

		y3, hidden = self.lstm1(y2, hidden)
		y4 = F.relu(self.fc1(y3))
		y5 = self.fc2(y4)
		y5 = y5.view(-1, 30, 50, 3)
		return y5, hidden

def videoGenerator(video, label, numEpochs, numFrames, sequenceSize):
	numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)
	for e in range(numEpochs):
		for j in range(numSequences):
			seqStartIndex, seqEndIndex = j * sequenceSize, np.minimum((j + 1) * sequenceSize, numFrames - 1)
			currentVideoSeq = video[seqStartIndex : seqEndIndex]
			currentLabelSeq = label[seqStartIndex : seqEndIndex]

			currentVideoSeq = np.expand_dims(minMaxNormalizeData(currentVideoSeq, 0, 255), axis=0)
			currentLabelSeq = np.expand_dims(minMaxNormalizeData(currentLabelSeq, 0, 255), axis=0)

			yield currentVideoSeq, currentLabelSeq

def minMaxNormalizeFrame(frame):
	Min, Max = np.min(frame), np.max(frame)
	frame -= Min
	frame /= (Max - Min)
	frame *= 255
	return np.uint8(frame)

def make_frame(t, model, video):
	global hiddenState
	# print(npFrame.dtype, npFrame.shape)
	t = int(t * 30)
	npFrame = video[t]
	plot_image(npFrame)
	npFrame = np.expand_dims(np.float32(npFrame), axis=0)
	trFrame = maybeCuda(tr.from_numpy(npFrame))

	# Reset hidden state every N values
	if t == 0:
		hiddenState = None

	trResult, hiddenState = model.forward(trFrame, hiddenState)
	npResult = trResult.detach().cpu().numpy()[0]
	npResult = minMaxNormalizeFrame(npResult)
	hiddenState.detach_()
	# print(npGetInfo(npResult))
	plot_image(npResult)
	show_plots()
	return npResult

def main():
	assert sys.argv[1] in ("train", "test", "retrain")

	video = pims.Video(sys.argv[2])
	label = pims.Video(sys.argv[3])
	assert len(video) == len(label)
	numFrames = len(video) // 2
	sequenceSize = 15
	model = maybeCuda(RecurrentCNN(inputSize=video.frame_shape, hiddenSize=10, outputSize=label.frame_shape))
	model.setCriterion(lambda x, y : tr.sum((x - y)**2))

	if sys.argv[1] == "train":
		assert len(sys.argv) == 4
		model.setOptimizer(optim.SGD, lr=0.0001, momentum=0.9)
		print(model.summary())

		generator = videoGenerator(video, label, numEpochs=10, numFrames=numFrames, sequenceSize=sequenceSize)
		numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)
		print("Video shape: %s. Label shape: %s. Num frames: %d" % (video.frame_shape, label.frame_shape, numFrames))

		model.train_generator(generator, stepsPerEpoch=numSequences, numEpochs=10, callbacks=[SaveModels(type="best")])
	elif sys.argv[1] == "test":
		assert len(sys.argv) == 5
		# Here label is used just to extract the correct shape
		model.load_weights(sys.argv[4])

		frameCallback = partial(make_frame, model=model, video=video)

		clip = mpy.VideoClip(frameCallback, duration=numFrames//30)
		clip.write_videofile("test_output.mp4", fps=30, verbose=False, progress_bar=True)

	# 	saveVideo(npData=storeFrames.result, fileName="video_result.mp4", fps=30)
	# elif sys.argv[1] == "retrain":
	# 	assert len(sys.argv) == 6
	# 	model.load_model(sys.argv[4])
	# 	model.setStartEpoch(int(sys.argv[5]))
	# 	print(model.summary())

	# 	generator = videoGenerator(video, label, numEpochs=10, numFrames=numFrames, sequenceSize=sequenceSize)
	# 	numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)
	# 	print("Video shape: %s. Label shape: %s. Num frames: %d" % (video.frame_shape, label.frame_shape, numFrames))

	# 	model.train_generator(generator, stepsPerEpoch=numSequences, numEpochs=10, callbacks=[SaveModels(type="all")])

if __name__ == "__main__":
	main()
