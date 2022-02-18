from typing import List

import torch
from torch import nn
from wenet.utils.huffman_tree import HuffmanTree
from wenet.utils.multiprocessing import ProcessPool, Worker
from queue import PriorityQueue
import time


class HSoftmaxLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        attention_dim: int,
        tree: HuffmanTree,
        num_workers: int,
    ):
        assert tree.info()['leaf_cnt'] == vocab_size
        super().__init__()

        self.tree = tree
        self.inner_vector = nn.Linear(attention_dim, tree.inner_cnt, False)
        self.register_buffer(
            'path_mask_sign', torch.Tensor(tree.path_mask_sign))
        self.register_buffer(
            'path_mask_bias', torch.Tensor(tree.path_mask_bias))

        self.num_workers = mp.cpu_count() if num_workers is None else num_workers
        self.pool = None
        self.queue = None

    def forward(self, att: torch.Tensor):
        h = self.inner_vector(att)
        h = torch.sigmoid(h)
        # h.size = [inner_cnt], path_mask.size = [vocab_size, inner_cnt]
        H = h * self.path_mask_sign
        # H.size = [vocab_size, inner_cnt]
        H = H + self.path_mask_bias
        return torch.sum(torch.log(H), -1)

    def beam_search(self, att: torch.Tensor, beam_size: int) -> List[List[int]]:
        if self.pool is None:
            self.inner_vector = self.inner_vector.to('cpu')
            self.pool = ProcessPool(
                self.num_workers, SearchWorker, self.inner_vector, self.tree.root, beam_size)
        att = att.to('cpu')
        logps, indexs = [], []
        # start = time.time()
        # # single process method
        # for t in att:
        #     top_k_logp, top_k_index = self.pool.workers[0](t)
        #     logps.append(top_k_logp)
        #     indexs.append(top_k_index)

        # accelarate ratio 2, shared memeory replace Queue may be faster
        for top_k_logp, top_k_index in self.pool.imap(att):
            logps.append(top_k_logp)
            indexs.append(top_k_index)
        # print('beam', time.time()-start)
        return torch.FloatTensor(logps), torch.LongTensor(indexs)  # (B*N, N)


class SearchWorker(Worker):
    def __init__(self, in_queue, out_queue, inner_vector, root, beam_size):
        super().__init__(in_queue, out_queue)
        self.inner_vector = inner_vector  # pickle these could be time consuming?
        self.root = root
        self.beam_size = beam_size

    def __call__(self, att, idx=None):
        assert att.dim() == 1  # batch size should be 1, att should an attention for only 1 token

        class Candidate:
            def __init__(self, prob, node):
                self.prob = prob
                self.node = node

            def __lt__(self, other):
                return self.prob < other.prob
        candidates = PriorityQueue()
        tokens = PriorityQueue()
        candidates.put(Candidate(1.0, self.root))

        while not candidates.empty():
            cur = candidates.get()

            if cur.node.token is not None:  # leaf node
                tokens.put(cur)
                while tokens.qsize() > self.beam_size:
                    tokens.get()
            else:  # inner node
                prob_left = torch.sigmoid(
                    torch.dot(att, self.inner_vector.weight[cur.node.idx])).item()
                prob_right = 1.0 - prob_left
                candidates.put(Candidate(prob_left, cur.node.left))
                candidates.put(Candidate(prob_right, cur.node.right))

                while candidates.qsize() > self.beam_size:
                    candidates.get()

        top_k_logp, top_k_index = [], []
        while not tokens.empty():
            cur = tokens.get()
            top_k_logp.append(cur.prob)
            top_k_index.append(cur.node.idx)

        return top_k_logp, top_k_index
