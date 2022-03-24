
import numpy as np
import torch
import torch.multiprocessing as mp
# import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Pipe
# from multiprocessing import Queue
# purpose of this class is to save function context to subprocess

import time


class Worker(Process):
    """Process executing tasks from a given args queue"""

    def __init__(self, workerid, run_flag,
                 jobid_share, lock, batch_size_share,
                 att_share,
                 tokenid_share, prob_share,
                 start_event, data_ready, data_ready_lock, data_ready_event):
        super().__init__()
        self.workerid = workerid
        self.lock = lock
        self.daemon = True
        self.jobid_share = jobid_share
        self.batch_size_share = batch_size_share
        self.att_share = att_share
        self.tokenid_share = tokenid_share
        self.prob_share = prob_share
        self.start_event = start_event
        self.data_ready = data_ready
        self.data_ready_lock = data_ready_lock
        self.data_ready_event = data_ready_event

    def __call__(self, att):  # For subclass to implement
        pass

    def run(self):
        while True:
            while True:
                with self.lock:
                    jobid = self.jobid_share.value
                    flag = jobid < self.batch_size_share.value
                    if flag:
                        self.jobid_share.value = jobid + 1
                    elif jobid == self.batch_size_share.value:
                        self.start_event.clear()
                if not flag:
                    self.start_event.wait()
                else:
                    break
            # while not self.run_flag.value:
            #     pass
            # jobid = self.workerid
            att_np = np.frombuffer(
                self.att_share[jobid].get_obj(), dtype=np.float32)
            att = torch.from_numpy(att_np)

            try:
                ret = self(att)

                self.tokenid_share[jobid].value = ret[1].item()
                self.prob_share[jobid].value = ret[0].item()
                with self.data_ready_lock:
                    self.data_ready.value += 1
                    # if self.data_ready.value == self.batch_size_share.value:
                    #     self.data_ready_event.set()
                # print(self.tokenid_share, self.prob_share)
                # print('put time', time.time() - start)
            except Exception as e:
                print(e)
            # self.finish_event.wait()
            # self.finish_event.clear()
            # while not self.finish_flag.value:
            #     pass
            # self.finish_flag.value = 0


class ProcessPool:
    """Pool of Process consuming args from a queue"""

    def __init__(self, num_workers, batch_size, attention_dim, Worker, *args):
        self.num_workers = num_workers
        self.att_share = [mp.Array('f', attention_dim)
                          for _ in range(batch_size)]
        self.att_np = [np.frombuffer(att.get_obj(), dtype=np.float32)
                       for att in self.att_share]
        self.tokenid_share = [mp.Value('l') for _ in range(batch_size)]
        self.prob_share = [mp.Value('f') for _ in range(batch_size)]

        self.jobid_share = mp.Value('l', -1)
        self.batch_size_share = mp.Value('l', -1)
        self.lock = mp.Lock()
        self.start_event = mp.Event()
        self.data_ready = mp.Value('l')
        self.data_ready_lock = mp.Lock()
        self.data_ready_event = mp.Event()
        # self.closed = False
        self.workers = []
        for i in range(num_workers):
            worker = Worker(i, self.jobid_share, self.lock, self.batch_size_share,
                            self.att_share,
                            self.tokenid_share, self.prob_share,
                            self.start_event, self.data_ready, self.data_ready_lock, self.data_ready_event,
                            * args)
            worker.start()
            self.workers.append(worker)

    def imap(self, att):
        batch_size = att.size()[0]
        for i in range(batch_size):
            np.copyto(self.att_np[i], att[i].numpy())

        self.data_ready.value = 0
        self.jobid_share.value = 0
        self.batch_size_share.value = batch_size
        self.start_event.set()

        # self.data_ready_event.wait()
        # self.data_ready_event.clear()
        while self.data_ready.value < batch_size:
            pass

        for i in range(batch_size):
            yield ([self.prob_share[i].value], [self.tokenid_share[i].value])

    def imap_unordered(self, iterable):
        num = 0
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1

        for i in range(num):
            yield self.out_queue.get()[1]
