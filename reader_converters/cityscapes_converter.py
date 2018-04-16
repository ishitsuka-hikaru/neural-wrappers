import os
import sys
import numpy as np
import cv2
import h5py
from PIL import Image
from lycon import resize, Interpolation
from argparse import ArgumentParser

# TODO: If Optical Flow is used, we need to import the algorithm that computes for each frame using the values at
#  rgb and rgb_next (or seq_rgb_X and seq_rgb_X_next)
# sys.path.append("")
# from inference import runImages

def flushPrint(message):
	sys.stdout.write(message + "\n")
	sys.stdout.flush()

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type", help="train/test/val")
	parser.add_argument("base_path", help="The path where all the subdirectories are extracted")
	parser.add_argument("output_file", help="The name of the output h5py file")
	parser.add_argument("--rgb", help="Do RGB (20th frame of each sequence)")
	parser.add_argument("--depth", help="Do Depth (20th frame of each sequence)")
	parser.add_argument("--optical_flow", help="Do Optical Flow (20th frame of each sequence)")
	parser.add_argument("--optical_flow_algorithm", help="What Optical Flow algorithm to use")
	parser.add_argument("--semantic", help="Do Semantic Fine (20th frame of each sequence)")
	parser.add_argument("--rgb_first_frame", help="Do RGB (1st frame of each sequence)")
	parser.add_argument("--seq_rgb", help="Do RGB (all 30 frames of each sequence)")
	parser.add_argument("--seq_depth", help="Do Depth (all 30 frames of each sequence)")
	parser.add_argument("--seq_optical_flow", help="Do Optical Flow (all 30 frames of each sequence)")
	parser.add_argument("--seq_skip_frames", help="How many frames to skip in sequence (must be 1 <= x <= 10)", \
		type=int, default=10)
	args = parser.parse_args()

	assert args.type in ("train", "test", "val")
	args.rgb = bool(args.rgb)
	args.depth = bool(args.depth)
	args.optical_flow = bool(args.optical_flow)
	args.semantic = bool(args.semantic)
	args.rgb_first_frame = bool(args.rgb_first_frame)
	args.seq_rgb = bool(args.seq_rgb)
	args.seq_depth = bool(args.seq_depth)
	args.seq_optical_flow = bool(args.seq_optical_flow)

	args.base_path = os.path.abspath(args.base_path)
	args.output_file = os.path.abspath(args.output_file)
	assert os.path.exists(args.base_path)

	if args.optical_flow:
		assert args.optical_flow_algorithm in ("flownet2s", )
	assert args.seq_skip_frames > 0 and args.seq_skip_frames <= 10
	return args

def getPaths(path, type, semanticConstraint=False):
	assert type in ("train", "test", "val")
	baseDir = path + os.sep + type
	os.chdir(baseDir)

	if semanticConstraint:
		condition = lambda x : x.endswith("png") and x.find("labelIds") != -1
	else:
		condition = lambda x : x.endswith("png")

	pngPaths = []
	allDir = sorted(os.listdir())
	for Dir in allDir:
		allPng = sorted(list(filter(condition, os.listdir(Dir))))
		allPng = list(map(lambda x : baseDir + os.sep + Dir + os.sep + x, allPng))
		pngPaths.extend(allPng)
	return pngPaths

# Compute the required png paths
def setup(args):
	paths = {}

	seqRgbPath = args.base_path + os.sep + "leftImg8bit_sequence"
	seqDepthPath = args.base_path + os.sep + "disparity_sequence"
	if args.rgb or args.rgb_first_frame or args.seq_rgb or args.seq_optical_flow or args.optical_flow:
		paths["seq_rgb"] = getPaths(seqRgbPath, args.type)
	if args.depth or args.seq_depth:
		paths["seq_depth"] = getPaths(seqDepthPath, args.type)

	if args.rgb or args.optical_flow:
		# RGB are each 20th frame (out of 30) from seq_rgb
		indexes = np.arange(len(paths["seq_rgb"]) // 30) * 30 + 19
		paths["rgb"] = []
		for index in indexes:
			paths["rgb"].append(paths["seq_rgb"][index])

	if args.rgb_first_frame:
		# RGB, but use just first frame (out of 30) from seq_rgb
		indexes = np.arange(len(paths["seq_rgb"]) // 30) * 30 + 0
		paths["rgb_first_frame"] = []
		for index in indexes:
			paths["rgb_first_frame"].append(paths["seq_rgb"][index])

	if args.depth:
		indexes = np.arange(len(paths["seq_depth"]) // 30) * 30 + 19
		paths["depth"] = []
		for index in indexes:
			paths["depth"].append(paths["seq_depth"][index])

	if args.semantic:
		semanticPath = args.base_path + os.sep + "gtFine"
		paths["semantic"] = getPaths(semanticPath, args.type, semanticConstraint=True)

	if args.seq_rgb or args.seq_optical_flow:
		indexes = np.arange(0, len(paths["seq_rgb"]), args.seq_skip_frames)
		# This is needed for optical flow, so the optical flow frames are from same sequence
		whereIndexes = np.where(indexes % 29 == 0)
		indexes[whereIndexes] -= 1
		keyName = "seq_rgb_%d" % (args.seq_skip_frames)
		paths[keyName] = []
		for index in indexes:
			paths[keyName].append(paths["seq_rgb"][index])

	if args.seq_depth:
		indexes = np.arange(0, len(paths["seq_depth"]), args.seq_skip_frames)
		# This is needed for optical flow, so the optical flow frames are from same sequence
		whereIndexes = np.where(indexes % 29 == 0)
		indexes[whereIndexes] -= 1
		keyName = "seq_depth_%d" % (args.seq_skip_frames)
		paths[keyName] = []
		for index in indexes:
			paths[keyName].append(paths["seq_depth"][index])

	if args.optical_flow:
		# Use 21st frame (next from rgb) to compute optical flow
		indexes = np.arange(len(paths["seq_rgb"]) // 30) * 30 + 20
		paths["rgb_next"] = []
		for index in indexes:
			paths["rgb_next"].append(paths["seq_rgb"][index])

	if args.seq_optical_flow:
		indexes = np.arange(0, len(paths["seq_rgb"]), args.seq_skip_frames)
		whereIndexes = np.where(indexes % 29 == 0)
		indexes[whereIndexes] -= 1
		indexes += 1
		keyName = "seq_rgb_%d_next" % (args.seq_skip_frames)
		paths[keyName] = []
		for index in indexes:
			paths[keyName].append(paths["seq_rgb"][index])

	if "seq_rgb" in paths:
		del paths["seq_rgb"]
	if "seq_depth" in paths:
		del paths["seq_depth"]

	return paths

def doDepth(depthPath):
	image = np.uint16(Image.open(depthPath)[30 : 900, 100 : 1920])
	whereZero = np.uint8((image == 0))
	image = cv2.inpaint(image, whereZero, inpaintRadius=7, flags=cv2.INPAINT_NS)
	return image

def doPng(imagePath):
	image = np.uint8(Image.open(imagePath))[30 : 900, 100 : 1920]
	return image

def doFlow(image, nextImage):
	# first upsample images to 1024x2048 (flownet2 accepts power of 2 inputs), then downsample the resulting flow
	#  back to original shape of 870x1820
	image = resize(image, height=1024, width=2048, interpolation=Interpolation.LANCZOS)
	nextImage = resize(nextImage, height=1024, width=2048, interpolation=Interpolation.LANCZOS)
	flow = np.zeros((870, 1820, 2), dtype=np.float32)
	flow_y, flow_x = runImages(image, nextImage)
	flow[..., 0] = resize(flow_y, height=870, width=1820, interpolation=Interpolation.LANCZOS)
	flow[..., 1] = resize(flow_x, height=870, width=1820, interpolation=Interpolation.LANCZOS)
	return flow

def prepareData(group, name, dataShape, dtype):
	if name in group:
		del group[name]
	group.create_dataset(name, shape=dataShape, dtype=dtype)

def doTheThingy(file, args, paths):
	if not args.type in file:
		group = file.create_group(args.type)
	else:
		group = file[args.type]

	if args.rgb:
		numData = len(paths["rgb"])
		flushPrint("Doing RGB (%d pictures)" % (numData))
		prepareData(group, name="rgb", dataShape=(numData, 870, 1820, 3), dtype=np.uint8)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("RGB %d/%d done." % (i, numData))
			group["rgb"][i] = doPng(paths["rgb"][i])

	if args.rgb_first_frame:
		numData = len(paths["rgb_first_frame"])
		flushPrint("Doing RGB first frame (%d pictures)" % (numData))
		prepareData(group, name="rgb_first_frame", dataShape=(numData, 870, 1820, 3), dtype=np.uint8)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("RGB first frame %d/%d done." % (i, numData))
			group["rgb_first_frame"][i] = doPng(paths["rgb_first_frame"][i])

	if args.depth:
		numData = len(paths["depth"])
		flushPrint("Doing Depth (%d pictures)" % (numData))
		prepareData(group, name="depth", dataShape=(numData, 870, 1820), dtype=np.uint16)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("Depth %d/%d done." % (i, numData))
			group["depth"][i] = doDepth(paths["depth"][i])

	if args.semantic:
		numData = len(paths["semantic"])
		flushPrint("Doing Semantic (%d pictures)" % (numData))
		prepareData(group, name="semantic", dataShape=(numData, 870, 1820), dtype=np.uint8)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("Semantic %d/%d done." % (i, numData))
			group["semantic"][i] = doDepth(paths["semantic"][i])

	if args.seq_rgb:
		keyName = "seq_rgb_%d" % (args.seq_skip_frames)
		numData = len(paths[keyName])
		flushPrint("Doing RGB sequence (%d pictures, %d skip frame)" % (numData, args.seq_skip_frames))
		prepareData(group, name=keyName, dataShape=(numData, 870, 1820), dtype=np.uint8)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("RGB sequence %d/%d done." % (i, numData))
			group[keyName][i] = doDepth(paths[keyName][i])

	if args.seq_depth:
		keyName = "seq_depth_%d" % (args.seq_skip_frames)
		numData = len(paths[keyName])
		flushPrint("Doing Depth sequence (%d pictures, %d skip frame)" % (numData, args.seq_skip_frames))
		prepareData(group, name=keyName, dataShape=(numData, 870, 1820), dtype=np.uint16)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("Depth sequence %d/%d done." % (i, numData))
			group[keyName][i] = doDepth(paths[keyName][i])

	if args.optical_flow:
		keyName = args.optical_flow_algorithm
		assert len(paths["rgb"]) == len(paths["rgb_next"])
		numData = len(paths["rgb"])
		flushPrint("Doing Optical Flow (%d pictures, %s algorithm)" % (numData, args.optical_flow_algorithm))
		prepareData(group, name=keyName, dataShape=(numData, 870, 1820, 2), dtype=np.float32)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("Optical Flow %d/%d done." % (i, numData))
			image1 = doPng(paths["rgb"][i])
			image2 = doPng(paths["rgb_next"][i])
			group[keyName][i] = doFlow(image1, image2)

	if args.seq_optical_flow:
		keyNameFlow = "seq_%s_%d" % (args.optical_flow_algorithm, args.seq_skip_frames)
		keyNameRGB1 = "seq_rgb_%d" % (args.seq_skip_frames)
		keyNameRGB2 = "seq_rgb_%d_next" % (args.seq_skip_frames)
		assert len(paths[keyNameRGB1]) == len(paths[keyNameRGB2])
		numData = len(paths[keyNameRGB1])
		flushPrint("Doing Optical Flow sequence (%d pictures, %s algorithm)" % (numData, args.optical_flow_algorithm))
		prepareData(group, name=keyNameFlow, dataShape=(numData, 870, 1820, 2), dtype=np.float32)
		for i in range(numData):
			if i % 10 == 0:
				flushPrint("Doing Optical Flow sequence %d/%d done." % (i, numData))
			image1 = doPng(paths[keyNameRGB1][i])
			image2 = doPng(paths[keyNameRGB2][i])
			group[keyNameFlow][i] = doFlow(image1, image2)

def main():
	args = getArgs()

	paths = setup(args)
	for key in paths:
		flushPrint("%s => %d images" % (key, len(paths[key])))

	file = h5py.File(args.output_file, "a")
	doTheThingy(file, args, paths)

if __name__ == "__main__":
	main()
