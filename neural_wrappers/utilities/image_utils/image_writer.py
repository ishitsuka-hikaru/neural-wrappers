def writeImageOpenCV(file, path):
	import cv2
	cv2.imwrite(path, file[..., ::-1])

def writeImagePIL(file, path):
	assert False, "TODO"

def writeImageLycon(file, path):
	import lycon
	lycon.save(path, file)

def tryWriteImage(file, path, count=5, imgLib="opencv"):
	assert imgLib in ("opencv", "PIL", "lycon")
	from ..np_utils import npGetInfo
	f = {
		"opencv" : writeImageOpenCV,
		"PIL" : writeImagePIL,
		"lycon" : writeImageLycon
	}[imgLib]

	i = 0
	while True:
		try:
			return f(file, path)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
