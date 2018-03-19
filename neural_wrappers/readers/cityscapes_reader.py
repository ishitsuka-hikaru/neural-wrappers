import h5py
import numpy as np
import os
import pims
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import resize_batch

class CityScapesReader(DatasetReader):
	# @param[in] datasetPath The path to the root directory of the CityScapes videos
	# @param[in] imageShape The shape to which the images are resized during the iteration phase. Frames have shape
	#  870x1820x3, but normal values are 512x1024x3 (some networks require power of 2 inputs).
	# @param[in] labelShape The shape of the depths (TOOD: a list including shape of flow as well). Frames have a shape
	#  of 870x1820, but normal values are 256x512, which is the network output for 512x1024x3 input.
	# @param[in] skipFrames Each video is consistent of multiple 30-frames scenes (1 second at 30fps). For videos, we
	# want to process every frame (so skipFrames is 1 by default). For some other type of processing (like training a
	#  non-recurrent neural network), we might want to skip some frames (like get only every nth frame), so this value
	#  is higher. It cannot exceed 30, because there are only 30 frames in every scene and each scene is an independent
	#  unit of work.
	# @param[in] transforms A list of transformations to be applied on each iteration.
	# @param[in] dataSplit A list/tuple that sums to 100 representing the split between train/test/validation sets.
	#  The split is done based on the videos (so for 80/0/20, 80% of videos are used for training, 0% for testing and
	#  20% for validation). This does not take into account the size of the videos. So the 80% of the videos could be
	#  much smaller in number of frames than the last 20%. Depends on the dataset. For CityScapes, most of videos are
	#  600 frames long, with some final videos of each scenes being smaller (varying from 30 to 570).
	# @param[in] precomputedDurations If set true, the totalDurations and durations arrays are pre-computed. This is
	#  only valid for a dataSplit of (80, 0, 20). For genericity, this should be False, but takes about 10 seconds
	#  when the object is instantiated.
	def __init__(self, datasetPath, imageShape, labelShape, skipFrames=1, transforms=["none"], dataSplit=(80, 0, 20),\
		precomputedDurations=False):
		assert skipFrames > 0 and skipFrames <= 30
		assert precomputedDurations == False or (precomputedDurations == True and dataSplit == (80, 0, 20))
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.skipFrames = skipFrames
		self.precomputedDurations = precomputedDurations

		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape)
		self.validationAugmenter = Transformer(["none"], dataShape=imageShape, labelShape=labelShape)
		self.setup()

	def setup(self):
		videosPath = self.datasetPath + os.sep + "video"
		depthPath = self.datasetPath + os.sep + "depth"
		flowFarnebackPath = self.datasetPath + os.sep + "flow_farneback"
		flowDeepFlow2Path = self.datasetPath + os.sep + "flow_deepflow2"

		allVideos = os.listdir(videosPath)
		allVideos = sorted(list(filter(lambda x : x.endswith("mp4"), allVideos)))
		numVideos = len(allVideos)
		assert numVideos > 0

		self.indexes, self.numData = self.computeIndexesSplit(numVideos)

		# Using the indexes, find all the paths to all the videos of all types for all labels. These will be used
		#  during iteration to actually read the videos and access the frame for training/testing.
		self.paths = {
			"train" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			},
			"test" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			},
			"validation" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			}
		}

		if self.precomputedDurations == True:
			self.totalDurations = {
				'train': 120480,
				'test': 0,
				'validation': 29520
			}

			self.durations = {
				'train': np.array([600, 600, 600, 600, 600, 600, 600, 600, 420, 600, 600, 600, 600, 600, \
					600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 120, 600, \
					600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 30, 600, \
					600, 600, 600, 480, 600, 600, 180, 600, 600, 600, 600, 600, 600, 600, 480, 600, 600, \
					600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 420, 600, 600, 600, \
					600, 150, 600, 600, 600, 30, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, \
					600, 600, 270, 600, 600, 600, 600, 600, 210, 600, 600, 600, 600, 600, 600, 600, 600, 600, \
					600, 600, 600, 240, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, \
					600, 600, 600, 480, 600, 600, 600, 600, 600, 570, 600, 600, 600, 600, 570, 600, 600, \
					540, 600, 600, 570, 600, 600, 600, 600, 600, 600, 540, 600, 600, 600, 600, 600, 600, \
					600, 600, 600, 600, 600, 600, 420, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, \
					600, 540, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, \
					600, 420, 600, 600, 600, 600]),
				'test': np.array([]),
				'validation': np.array([600, 600, 600, 600, 600, 600, 150, 600, 600, 600, 600, 600, 600, \
					600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 480, 600, 600, 600, 600, 600, \
					600, 600, 120, 600, 600, 600, 600, 450, 600, 600, 600, 600, 600, 600, 600, 60, 600, \
					600, 600, 600, 600, 600, 60])
			}
		else:
			self.totalDurations = {
				"train" : 0,
				"test" : 0,
				"validation" : 0
			}

			self.durations = {
				"train" : None,
				"test" : None,
				"validation" : None
			}

		for Type in ["train", "test", "validation"]:
			typeVideos = allVideos[self.indexes[Type][0] : self.indexes[Type][1]]
			if self.precomputedDurations == False:
				self.durations[Type] = np.zeros((len(typeVideos), ), dtype=np.int32)
			for i in range(len(typeVideos)):
				videoName = typeVideos[i]
				depthName = depthPath + os.sep + "depth_" + videoName[6 : ]
				flowFarnebackName = flowFarnebackPath + os.sep + "flow_farneback_" + videoName[6 : -4] + ".avi"
				flowDeepFlow2Name = flowDeepFlow2Path + os.sep + "flow_deepflow2_" + videoName[6 : -4] + ".avi"
				videoName = videosPath + os.sep + videoName

				self.paths[Type]["video"].append(videoName)
				self.paths[Type]["depth"].append(depthName)
				self.paths[Type]["flow_farnaback"].append(flowFarnebackName)
				self.paths[Type]["flow_deepflow2"].append(videoName)

				# Also save the duration, so we can compute the number of iterations (Takes about 10s for all dataset)
				if self.precomputedDurations == False:
					video = pims.Video(videoName)
					# Some videos have 601 frames, instead of 600, so remove that last one (also during iteration)
					videoLen = len(video) - (len(video) % 30)
					self.totalDurations[Type] += videoLen
					self.durations[Type][i] = videoLen

		# print(self.durations)
		print(("[CityScapes Reader] Setup complete. Num videos: %d. Train: %d, Test: %d, Validation: %d. " + \
			"Frame shape: %s. Labels shape: %s. Frames: Train: %d, Test: %d, Validation: %d.") % (numVideos, \
			self.numData["train"], self.numData["test"], self.numData["validation"], self.imageShape, \
			self.labelShape, self.totalDurations["train"], self.totalDurations["test"], \
			self.totalDurations["validation"]))

	def getNumIterations(self, type, miniBatchSize, accountTransforms=False):
		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		numIterations = 0
		thisDurations = self.durations[type]
		for i in range(N):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			thisVideosDurations = thisDurations[startIndex : endIndex]
			# [30, 60, 450, 420, 60] => 450 iterations for this batch, but the other elements will only contribute
			#  for their amount (so 30, 60, 420, 60). Thus, for first 30 iterations we get 5 items from this batch,
			#  then until 60 we get 4 items, then until 420 we get 2 items and for 420-450 we just get 1 item.
			maxDuration = np.max(thisVideosDurations)
			batchContribution = int(np.ceil(maxDuration / self.skipFrames))
			numIterations += batchContribution
		return numIterations if accountTransforms == False else numIterations * len(self.transforms)

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		thisPaths = self.paths[type]
		thisDurations = self.durations[type]

		for i in range(N):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			videos = [pims.Video(thisPaths["video"][j]) for j in range(startIndex, endIndex)]
			depth_videos = [pims.Video(thisPaths["depth"][j]) for j in range(startIndex, endIndex)]
			# flow_farneback_videos = [pims.Video(thisPaths["flow_farnaback"][j]) for j in range(startIndex, endIndex)]
			# flow_deepflow2_videos = [pims.Video(thisPaths["flow_deepflow2"][j]) for j in range(startIndex, endIndex)]

			numVideos = len(videos)
			frameShape = videos[0].frame_shape
			images = np.zeros((numVideos, *frameShape))
			# Depths are 3 grayscale identical channels (due to necessity of saving them as RGB)
			depths = np.zeros((numVideos, *frameShape[0 : 2]))
			# flow_farnaback, flow_deepflow2 = ...

			thisVideosDurations = thisDurations[startIndex : endIndex]
			maxDuration = np.max(thisVideosDurations)
			batchContribution = int(np.ceil(maxDuration / self.skipFrames))

			for frame_i in range(0, maxDuration, self.skipFrames):
				validIndex = 0
				for video_i in range(numVideos):
					if thisVideosDurations[video_i] <= frame_i:
						continue

					images[validIndex] = videos[video_i][frame_i] / 255
					depths[validIndex] = depth_videos[video_i][frame_i][..., 0] / 255
					validIndex += 1

				transformGenerator = augmenter.applyTransforms(images[0 : validIndex], \
					depths[0 : validIndex], "bilinear")
				for augmentedImages, augmentedDepths in transformGenerator:
					yield np.float32(augmentedImages), np.float32(augmentedDepths)
					del augmentedImages, augmentedDepths
			del videos
