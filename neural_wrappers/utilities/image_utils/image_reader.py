import numpy as np

def readImageOpenCV(path):
	import cv2
	bgr_image = cv2.imread(path)
	b, g, r = cv2.split(bgr_image)
	image = cv2.merge([r, g, b]).astype(np.uint8)
	return image

def readImagePIL(path):
	from PIL import Image
	image = np.array(Image.open(path), dtype=np.uint8)[..., 0 : 3]
	return image

def readImageLycon(path):
	from lycon import load
	image = load(path)[..., 0 : 3].astype(np.uint8)
	return image

def tryReadImage(path, count=5, imgLib="opencv"):
	assert imgLib in ("opencv", "PIL", "lycon")
	f = {
		"opencv" : readImageOpenCV,
		"PIL" : readImagePIL,
		"pillow" : readImagePIL,
		"lycon" : readImageLycon
	}[imgLib]

	i = 0
	while True:
		try:
			return f(path)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception