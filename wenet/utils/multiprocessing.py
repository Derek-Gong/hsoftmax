
import multiprocessing as mp
from multiprocessing import Process, Queue
# purpose of this class is to transfer function context to subprocess
class Worker(Process):
    """Process executing tasks from a given args queue"""
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True

    def __call__(self): # For subclass to implement
        pass

    def run(self):
        while True:
            i, args = self.in_queue.get()
            try:
                self.out_queue.put((i, self(*args)))
            except Exception as e:
                print(e)

class ProcessPool:
    """Pool of Process consuming args from a queue"""
    def __init__(self, num_workers, Worker, *args):
        self.in_queue = Queue()
        self.out_queue = Queue()
        # self.closed = False
        self.workers = []
        for _ in range(num_workers):
            self.workers.append(Worker(self.in_queue, self.out_queue, *args))

        for worker in self.workers:
            worker.start()
    
    def imap(self, iterable):
        num = 0
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1

        result = []
        for i in range(num):
            result.append(self.out_queue.get())
        result = sorted(result, key = lambda x: x[0])

        for res in result:
            yield res[1]

    def imap_unordered(self, iterable):
        num = 0
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1
        
        for i in range(num):
            yield self.out_queue.get()[1]
