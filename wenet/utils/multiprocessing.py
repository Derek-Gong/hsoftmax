
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

    def __init__(self, workerid, out_queue,
                 jobid_share,
                 att_share,
                 tokenid_share, prob_share,
                 start_flag, finish_flag,
                 start_event, finish_event):
        super().__init__()
        self.workerid = workerid
        self.out_queue = out_queue
        self.daemon = True
        self.jobid_share = jobid_share
        self.att_share = att_share
        self.tokenid_share = tokenid_share
        self.prob_share = prob_share
        self.start_event = start_event
        self.finish_event = finish_event
        self.start_flag = start_flag
        self.finish_flag = finish_flag

    def __call__(self, att):  # For subclass to implement
        pass

    def run(self):
        while True:
            start = time.time()
            # i, args = self.in_queue.get()
            # att = self.in_pipe.recv()
            # self.start_event.wait()
            self.start_event.recv()
            # self.start_event.clear()
            # while not self.start_flag.value:
            #     pass
            # self.start_flag.value = 0
            att_np = np.frombuffer(self.att_share.get_obj(), dtype=np.float32)
            att = torch.from_numpy(att_np)
            # i = self.jobid_share
            # att = attention_share
            # print('worker get ', time.time() - start)
            # print(i, args)
            try:
                start = time.time()
                ret = self(att)
                # res = (i, self(*args))
                # print('call time', time.time() - start)
                start = time.time()
                # self.out_queue.put(res)
                # self.out_pipe.send(res)
                self.tokenid_share.value = ret[1].item()
                self.prob_share.value = ret[0].item()
                # print(self.tokenid_share, self.prob_share)
                self.out_queue.put(self.workerid)
                # print('put time', time.time() - start)
            except Exception as e:
                print(e)
            self.finish_event.recv()
            # self.finish_event.wait()
            # self.finish_event.clear()
            # while not self.finish_flag.value:
            #     pass
            # self.finish_flag.value = 0


class ProcessPool:
    """Pool of Process consuming args from a queue"""

    def __init__(self, num_workers, attention_dim, Worker, *args):
        self.num_workers = num_workers
        self.att_share = [mp.Array('f', attention_dim)
                          for _ in range(num_workers)]
        self.att_np = [np.frombuffer(att.get_obj(), dtype=np.float32)
                       for att in self.att_share]
        self.jobid_share = [mp.Value('l') for _ in range(num_workers)]
        self.tokenid_share = [mp.Value('l') for _ in range(num_workers)]
        self.prob_share = [mp.Value('f') for _ in range(num_workers)]
        # self.start_event = [mp.Event() for _ in range(num_workers)]
        # self.finish_event = [mp.Event() for _ in range(num_workers)]
        self.start_event = [mp.Pipe() for _ in range(num_workers)]
        self.finish_event = [mp.Pipe() for _ in range(num_workers)]
        self.start_flag = [mp.Value('b', 0) for _ in range(num_workers)]
        self.finish_flag = [mp.Value('b', 0) for _ in range(num_workers)]

        self.out_queue = Queue()

        self.idle_workers = []
        # self.closed = False
        self.workers = []
        for i in range(num_workers):
            worker = Worker(i, self.out_queue,
                            self.jobid_share[i],
                            self.att_share[i],
                            self.tokenid_share[i], self.prob_share[i],
                            self.start_flag[i], self.finish_flag[i],
                            self.start_event[i][1], self.finish_event[i][1],
                            * args)
            worker.start()
            self.workers.append(worker)
            self.idle_workers.append(i)

    def imap(self, att):
        batch_size = att.size()[0]
        i = 0
        finish_cnt = 0
        result = [0] * batch_size
        start = time.time()

        def wait_finish_one():
            workerid = self.out_queue.get()
            ret = ([self.prob_share[workerid].value],
                   [self.tokenid_share[workerid].value])
            result[self.jobid_share[workerid].value] = ret
            # self.finish_event[workerid].set()
            self.finish_event[workerid][0].send(True)
            self.finish_flag[workerid].value = 1
            self.idle_workers.append(workerid)

        while i < batch_size:
            if len(self.idle_workers) > 0:
                workerid = self.idle_workers.pop(0)
                self.jobid_share[workerid].value = i
                np.copyto(self.att_np[workerid], att[i].numpy())
                # self.in_pipes[workerid][0].send(att[i])
                self.start_event[workerid][0].send(True)
                # self.start_event[workerid].set()
                self.start_flag[workerid].value = 1
                i += 1
            else:
                wait_finish_one()
                finish_cnt += 1
        while finish_cnt < batch_size:
            wait_finish_one()
            finish_cnt += 1
        # print('imap put ', time.time() - start)
        i = 0
        for res in result:
            yield res

    def imap_unordered(self, iterable):
        num = 0
        for i, args in enumerate(iterable):
            self.in_queue.put((i, (args,)))
            num += 1

        for i in range(num):
            yield self.out_queue.get()[1]
