from typing import List

import torch
from torch import nn
from wenet.utils.huffman_tree import HuffmanTree
from wenet.utils.multiprocessing import ProcessPool, Worker
from queue import PriorityQueue
import time


class HSoftmaxLayer(nn.Module):
    # def __init__(
    #     self,
    #     vocab_size: int,
    #     attention_dim: int,
    #     tree: HuffmanTree,
    #     num_workers: int,
    # ):
    #     assert tree.info()['leaf_cnt'] == vocab_size
    #     super().__init__()

    #     self.tree = tree
    #     self.inner_vector = nn.Linear(attention_dim, tree.inner_cnt, False)
    #     self.register_buffer(
    #         'path_mask_sign', torch.CharTensor(tree.path_mask_sign))
    #     self.register_buffer(
    #         'path_mask_bias', torch.CharTensor(tree.path_mask_bias))

    #     self.num_workers = mp.cpu_count() if num_workers is None else num_workers
    #     self.pool = None
    #     self.queue = None

    # def forward(self, att: torch.Tensor):
    #     # start = time.time()
    #     h = self.inner_vector(att)
    #     h = torch.sigmoid(h)
    #     h = h.unsqueeze(-2)
    #     # h.size = [batch, length, 1, inner_cnt], path_mask.size = [vocab_size, inner_cnt]
    #     H = h * self.path_mask_sign
    #     # H.size = [batch, length, vocab_size, inner_cnt]
    #     H = H + self.path_mask_bias
    #     # print('forward: ', time.time()-start)
    #     return torch.sum(torch.log(H), -1)

    def __init__(
        self,
        vocab_size: int,
        attention_dim: int,
        tree: HuffmanTree,
        num_workers: int,
    ):
        assert tree.info()['leaf_cnt'] == vocab_size
        super().__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.inner_vector = nn.Linear(attention_dim, tree.inner_cnt, False)
        self.tree = tree
        self.tree_depth = tree.depth

        self.register_buffer(
            'path_index', torch.tensor(tree.path_index, requires_grad=False).view(-1))  # [vocab_size * tree.depth]
        self.register_buffer(
            'path_sign', torch.tensor(tree.path_sign, requires_grad=False).view(-1))
        self.register_buffer(
            'path_bias', torch.tensor(tree.path_bias, requires_grad=False).view(-1))

        self.num_workers = mp.cpu_count() if num_workers is None else num_workers
        self.pool = None
        self.queue = None

        self.search_emb = None
        self.son_index = None

    def forward(self, att: torch.Tensor):
        # start = time.time()
        h = self.inner_vector(att)
        # h.size = [batch, length, inner_cnt]
        h = torch.sigmoid(h)
        h = torch.index_select(h, -1, self.path_index)
        # h.size = [batch, length, vocab_size * tree.depth]
        h = h * self.path_sign
        h = h + self.path_bias
        H = h.view(h.size()[0], -1, self.vocab_size, self.tree_depth)
        # print('forward: ', time.time()-start)
        return torch.sum(torch.log(H), -1)

    # def greedy_search(self, att: torch.Tensor) -> List[List[int]]:
    #     if self.search_emb = None:
    #         self.search_emb = torch.stack([self.inner_vector.weight, torch.zeros((self.vocab_size, self.tree.))])

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
