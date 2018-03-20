from threading import Thread, Condition

class ProducerThread(Thread):
	def __init__(self, condition, items, maxPrefetch, generator):
		Thread.__init__(self)
		self.condition = condition
		self.items = items
		self.maxPrefetch = maxPrefetch
		self.generator = generator
		self.endCondition = False

	def produceItem(self):
		try:
			item = next(self.generator)
		except StopIteration:
			item = None
		if self.endCondition == True:
			item = None
		return item

	def run(self):
		while True:
			self.condition.acquire()

			if self.endCondition == True:
				item = None
			else:
				if len(self.items) == self.maxPrefetch:
					# print("[Producer] List is full, waiting for consumer to consume.")
					self.condition.wait()
					# print("[Producer] Consumer consumed, producing now and notified.")
				item = self.produceItem()

			self.items.append(item)
			self.condition.notify()

			self.condition.release()

			if item == None:
				break

class PrefetchGenerator:
	def __init__(self, generator, maxPrefetch):
		self.items = []
		self.condition = Condition()
		self.producerThread = ProducerThread(self.condition, self.items, maxPrefetch, generator)
		self.producerThread.start()

	def __next__(self):
		self.condition.acquire()

		# print("[Consumer] Consuming...")
		if len(self.items) == 0:
			# print("[Consumer] List is empty, waiting for something to be produced.")
			self.condition.wait()
			# print("[Consumer] Producer added something to the list and notified.")
		item = self.items.pop(0)
		# print("Consumed", item)
		self.condition.notify()

		self.condition.release()

		if item == None:
			# print("[Consumer] Received None, finishing job")
			self.producerThread.join()
			raise StopIteration
		return item

	def __iter__(self):
		return self

	def __del__(self):
		self.producerThread.endCondition = True
		self.producerThread.join()
