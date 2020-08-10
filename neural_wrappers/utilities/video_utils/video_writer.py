from tqdm import tqdm

def writeVideoImageIo(file, path, fps):
	import imageio
	writer = imageio.get_writer(path, fps=fps)
	N = len(file)

	for i in tqdm(range(N)):
		writer.append_data(file[i])
	writer.close()

def tryWriteVideo(file, path, fps, count=5, imgLib="imageio"):
	assert imgLib in ("imageio", )
	f = {
		"imageio" : writeVideoImageIo
	}[imgLib]

	i = 0
	while True:
		try:
			return f(file, path, fps)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
