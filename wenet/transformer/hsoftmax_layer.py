from multiprocessing import dummy
from typing import List

import torch
from torch import nn
from wenet.utils.huffman_tree import HuffmanTree
from wenet.utils.multiprocessing import ProcessPool, Worker
from queue import PriorityQueue
import time
import logging


class HSoftmaxLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        attention_dim: int,
        huffman_tree_dir: str,
        num_workers: int,
        beam_size: int = 10
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.beam_size = beam_size
        self.eps = 1e-9

        self.tree = HuffmanTree.load(huffman_tree_dir)
        assert self.tree.info()['leaf_cnt'] == vocab_size

        self.inner_vector = nn.Linear(attention_dim, self.tree.inner_cnt, False)
        self.tree_depth = self.tree.depth

        self.register_buffer(
            'path_index', torch.tensor(self.tree.path_index, requires_grad=False).view(-1))  # [vocab_size * tree.depth]
        self.register_buffer(
            'path_sign', torch.tensor(self.tree.path_sign, requires_grad=False).view(-1))
        self.register_buffer(
            'path_bias', torch.tensor(self.tree.path_bias, requires_grad=False).view(-1))

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
        # avoid the case sigmoid(h) == 0 or 1 when |h| > 10
        H = torch.clip(H, self.eps, 1 - self.eps)
        logp = torch.sum(torch.log(H), -1)
        return logp

    def __init_search(self):
        if self.search_emb is None:
            device = self.inner_vector.weight.device
            self.search_emb = torch.cat([self.inner_vector.weight, torch.zeros(
                (self.vocab_size, self.attention_dim), device=device)], dim=0)
            self.son_index = torch.zeros(
                (self.tree.inner_cnt + self.vocab_size, 2), device=device, dtype=torch.int32) - 1

            def dfs(node):
                if node.tokenid is not None:
                    self.son_index[node.idx] = torch.tensor(
                        [node.idx, node.idx], device=device, dtype=torch.int32)
                    return
                self.son_index[node.idx] = torch.tensor(
                    [node.left.idx, node.right.idx], device=device, dtype=torch.int32)
                dfs(node.left)
                dfs(node.right)
            dfs(self.tree.root)

    # def greedy_search(self, att: torch.Tensor):
    #     self.__init_greedy_search()
        # batch_index = torch.zeros(
        #     (att.size()[0],), dtype=torch.int32, device=att.device)
        # batch_prob = torch.zeros((att.size()[0],), device=att.device)

    #     while not torch.all(torch.ge(batch_index, self.tree.inner_cnt)):
    @torch.no_grad()
    def beam_search(self, att: torch.Tensor, out_beam_size: int):
        self.__init_search()
        batch_size = att.size()[0]
        attention_dim = att.size()[-1]
        # [batch_size, attention_dim] -> [batch_size, attention_dim, 1]
        att = att.unsqueeze(-1)
        # [batch_size, beam_size]
        batch_index = torch.zeros(
            (batch_size, 1), dtype=torch.int32, device=att.device)
        batch_emb = torch.index_select(
            self.search_emb, 0, batch_index.view(-1)).view(batch_size, -1, attention_dim)
        batch_prob = torch.ones((batch_size, 1), device=att.device)
        # while index not all >= inner_cnt do:
        # dot & sigmoid (left prob) -> correct leaf prob to 1 -> stack right prob -> mul path prob -> topk -> gather
        while not torch.all(batch_index >= self.tree.inner_cnt):
            # bmm as a faster way to do batch dot
            # [batch_size, beam_size]
            left_prob = torch.sigmoid(
                torch.bmm(batch_emb, att).view(batch_size, -1))
            # print(left_prob.size(), batch_emb.size(), att.size())
            # correct leaf prob to 1, otherwise for all leaves left_prob = right_prob = 0.5
            left_prob[batch_index >= self.tree.inner_cnt] = 1.
            right_prob = 1 - left_prob
            # [batch_size, beam_size, 2]
            node_prob = torch.stack([left_prob, right_prob], dim=-1)
            cand_prob = batch_prob.unsqueeze(-1) * node_prob
            # [batch_size, beam_size * 2]
            cand_prob = cand_prob.view((batch_size, -1))
            # [batch_size, beam_size]
            cand_size = 2 * batch_index.size()[-1]
            k = min(self.beam_size, cand_size)
            batch_prob, top_id = cand_prob.topk(k)

            # [batch_size, beam_size * 2]
            cand_id = torch.index_select(
                self.son_index, 0, batch_index.view(-1)).view(batch_size, -1)
            # [batch_size, beam_size]
            batch_index = torch.gather(cand_id, -1, top_id)
            batch_emb = torch.index_select(
                self.search_emb, 0, batch_index.view(-1)).view(batch_size, -1, attention_dim)

        batch_index -= self.tree.inner_cnt
        # batch_prob = torch.clip(batch_prob, eps, 1)
        cand_size = batch_index.size()[-1]
        if out_beam_size > cand_size:
            dummy_index = torch.zeros(
                (batch_size, out_beam_size - cand_size), dtype=torch.int32, device=batch_index.device)
            # dummy_prob = self.eps * \
            #     torch.ones((batch_size, beam_size - cand_size),
            #                device=batch_prob.device)
            dummy_prob = torch.zeros(
                (batch_size, out_beam_size - cand_size), device=batch_prob.device)

            torch.cat([batch_index, dummy_index], dim=-1)
            torch.cat([batch_prob, dummy_prob], dim=-1)

        return torch.log(batch_prob), batch_index

    # def beam_search(self, att: torch.Tensor, beam_size: int):
    #     if self.pool is None:
    #         self.inner_vector=self.inner_vector.to('cpu')
    #         self.pool=ProcessPool(
    #             self.num_workers, SearchWorker, self.inner_vector, self.tree.root, beam_size)
    #     att=att.to('cpu')
    #     logps, indexs=[], []
    #     # start = time.time()
    #     # # single process method
    #     # for t in att:
    #     #     top_k_logp, top_k_index = self.pool.workers[0](t)
    #     #     logps.append(top_k_logp)
    #     #     indexs.append(top_k_index)

    #     # accelarate ratio 2, shared memeory replace Queue may be faster
    #     for top_k_logp, top_k_index in self.pool.imap(att):
    #         logps.append(top_k_logp)
    #         indexs.append(top_k_index)
    #     # print('beam', time.time()-start)
    #     return torch.FloatTensor(logps), torch.LongTensor(indexs)  # (B*N, N)


class SearchWorker(Worker):
    def __init__(self, in_queue, out_queue, inner_vector, root, beam_size):
        super().__init__(in_queue, out_queue)
        self.inner_vector = inner_vector  # pickle these could be time consuming?
        self.root = root
        self.beam_size = beam_size

    def __call__(self, att, idx=None):
        assert att.dim() == 1  # batch size should be 1, att should be an attention for only 1 token

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
