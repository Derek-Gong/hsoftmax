
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
# from multiprocessing import Queue
# purpose of this class is to save function context to subprocess

import time


class Worker(Process):
    """Process executing tasks from a given args queue"""

    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True
        # self.in_pipe = in_pipe

    def __call__(self, *args):  # For subclass to implement
        pass

    def run(self):
        while True:
            start = time.time()
            i, args = self.in_queue.get()
            print('worker get ', time.time() - start)
            # print(i, args)
            try:
                start = time.time()
                res = (i, self(*args))
                print('call time', time.time() - start)
                start = time.time()
                self.out_queue.put(res)
                print('put time', time.time() - start)
            except Exception as e:
                print(e)


class ProcessPool:
    """Pool of Process consuming args from a queue"""

    def __init__(self, num_workers, Worker, *args, **kwargs):
        self.in_queue = Queue()
        self.out_queue = Queue()
        # self.closed = False
        self.workers = []
        for _ in range(num_workers):
            worker = Worker(self.in_queue, self.out_queue, *args)
            worker.start()
            self.workers.append(worker)

    def imap(self, iterable):
        num = 0
        start = time.time()
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1
        print('imap put ', time.time() - start)
        result = []
        for i in range(num):
            result.append(self.out_queue.get())
        result = sorted(result, key=lambda x: x[0])

        for res in result:
            yield res[1]

    def imap_unordered(self, iterable):
        num = 0
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1

        for i in range(num):
            yield self.out_queue.get()[1]
