from multiprocessing import dummy
from typing import List

import torch
from torch import nn
import math
from wenet.utils.huffman_tree import HuffmanTree
from wenet.utils.multiprocessing import ProcessPool, Worker
from queue import PriorityQueue
import heapq
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
        self.inf = 1e18
        self.multilayer_decoding = 2
        self.multilayer_leaves = 2**self.multilayer_decoding
        self.multilayer_nodes = self.multilayer_decoding * self.multilayer_leaves

        self.tree = HuffmanTree.load(huffman_tree_dir)
        assert self.tree.info()['leaf_cnt'] == vocab_size

        self.inner_vector = nn.Linear(
            attention_dim, self.tree.inner_cnt, False)
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

    def beam_search(self, att: torch.Tensor, out_beam_size: int):
        # att = att.to('cpu')
        self.__init_search()
        if self.inner_vector.weight.is_cuda:
            return self.beam_search_gpu(att, out_beam_size)
        else:
            return self.beam_search_cpu(att, out_beam_size)

    #  core line: while not torch.all(torch.ge(batch_index, self.tree.inner_cnt)):
    def beam_search_gpu(self, att: torch.Tensor, out_beam_size: int):
        # print(torch.linalg.norm(att, ord=1, dim=-1))
        leaf_prone = min(self.beam_size, self.multilayer_leaves)
        batch_size = att.size()[0]
        # [batch_size, attention_dim] -> [batch_size, attention_dim, 1]
        att = att.unsqueeze(-1)
        # [batch_size, beam_size]
        batch_index = torch.zeros(
            (batch_size, 1), dtype=torch.int32, device=att.device)
        batch_emb = torch.index_select(
            self.search_emb, 0, batch_index.view(-1)).view(batch_size, -1, self.attention_dim)
        batch_prob = torch.ones((batch_size, 1), device=att.device)
        # while index not all >= inner_cnt do:
        # dot & sigmoid (node_prob) -> prod subtree prob to leaf -> mul path prob -> topk -> gather
        while not torch.all(batch_index >= self.tree.inner_cnt):
            # bmm as a faster way to do batch dot
            # [batch_size, beam_size]
            node_prob = torch.sigmoid(
                torch.bmm(batch_emb, att).view(batch_size, -1, self.multilayer_leaves, self.multilayer_decoding))
            # [batch_size, beam_size, multilayer_leaves]
            node_prob = torch.prod(node_prob, dim=-1)
            # prone leaf to beam_size
            node_prob, leaf_id = node_prob.topk(leaf_prone)
            leaf_id = leaf_id.view(batch_size, -1)
            # [batch_size, beam_size, beam_size]
            cand_prob = batch_prob.unsqueeze(-1) * node_prob
            # [batch_size, beam_size * beam_size]
            cand_prob = cand_prob.view(batch_size, -1)
            # [batch_size, beam_size]
            cand_size = self.multilayer_leaves * batch_index.size()[-1]
            k = min(self.beam_size, cand_size)
            batch_prob, top_id = cand_prob.topk(k)
            top_id = torch.gather(leaf_id, -1, top_id)
            # [batch_size, beam_size * multilayer_leaves]
            cand_id = torch.index_select(
                self.son_index, 0, batch_index.view(-1)).view(batch_size, -1)
            # [batch_size, beam_size]
            batch_index = torch.gather(cand_id, -1, top_id)
            batch_emb = torch.index_select(
                self.search_emb, 0, batch_index.view(-1)).view(batch_size, -1, self.attention_dim)

        batch_index -= self.tree.inner_cnt

        batch_prob, batch_index = self.__fill_beam(
            batch_prob, batch_index, out_beam_size)

        return torch.log(batch_prob), batch_index

    def __init_search(self):
        if self.search_emb is None:
            device = self.inner_vector.weight.device
            inf = self.inf * torch.ones(self.attention_dim, device=device)
            # self.search_emb = torch.cat([self.inner_vector.weight, inf * torch.ones(
            #     (self.vocab_size, self.attention_dim), device=device)], dim=0)
            self.search_emb = torch.zeros(
                self.tree.inner_cnt + self.vocab_size, self.multilayer_nodes, self.attention_dim, device=device)
            self.son_index = torch.zeros(
                (self.tree.inner_cnt + self.vocab_size, self.multilayer_leaves), device=device, dtype=torch.int32) - 1

            def dfs(anc):
                if anc is None:
                    return
                son_index = []
                subtree_embs = []

                def findson(node, embs, depth=0):
                    if depth == self.multilayer_decoding:
                        son_index.append(node.idx)
                        subtree_embs.append(torch.stack(embs))
                        return
                    if node.tokenid is None:
                        embs.append(self.inner_vector.weight[node.idx])
                        findson(node.left, embs, depth+1)
                        embs[-1] = -embs[-1]
                        findson(node.right, embs, depth+1)
                    else:
                        embs.append(inf)
                        findson(node, embs, depth+1)
                        embs[-1] = -embs[-1]
                        findson(node, embs, depth+1)
                    embs.pop()

                findson(anc, [])
                self.son_index[anc.idx] = torch.tensor(
                    son_index, device=device, dtype=torch.int32)
                self.search_emb[anc.idx] = torch.cat(subtree_embs)
                dfs(anc.left)
                dfs(anc.right)
            dfs(self.tree.root)

    def __fill_beam(self, batch_prob, batch_index, out_beam_size):
        batch_size = batch_index.size()[0]
        cand_size = batch_index.size()[-1]
        if out_beam_size > cand_size:
            dummy_index = torch.zeros(
                (batch_size, out_beam_size - cand_size), dtype=torch.int32, device=batch_index.device)
            dummy_prob = torch.zeros(
                (batch_size, out_beam_size - cand_size), device=batch_prob.device)

            batch_index = torch.cat([batch_index, dummy_index], dim=-1)
            batch_prob = torch.cat([batch_prob, dummy_prob], dim=-1)
        return batch_prob, batch_index

    # to-do: profile in_queue, out_queue: queue.get consume over 90% time of beam_search
    # use torch.multiprocessing.Process for gpu and shared memory for tensor
    # include gpu algorithm
    def beam_search_cpu(self, att: torch.Tensor, out_beam_size: int):
        if self.pool is None:
            self.inner_vector = self.inner_vector.to('cpu')
            self.pool = ProcessPool(
                self.num_workers, SearchWorker, self.inner_vector.weight, self.tree.root, self.beam_size,
                self.tree.inner_cnt, self.multilayer_leaves, self.search_emb, self.son_index)
        att = att.to('cpu')
        att = att.share_memory_()
        probs, indexs = [], []
        # # single process method
        # for t in att:
        #     top_k_logp, top_k_index = self.pool.workers[0](t)
        #     logps.append(top_k_logp)
        #     indexs.append(top_k_index)

        # accelarate ratio 2, shared memeory replace Queue may be faster
        for top_k_prob, top_k_index in self.pool.imap(att):
            probs.append(top_k_prob)
            indexs.append(top_k_index)

        batch_prob = torch.FloatTensor(probs)
        batch_index = torch.IntTensor(indexs)
        batch_prob, batch_index = self.__fill_beam(
            batch_prob, batch_index, out_beam_size)

        return torch.log(batch_prob), batch_index  # (B*N, N)


class SearchWorker(Worker):
    def __init__(self, in_queue, out_queue, inner_vector, root, beam_size,
                 inner_cnt, multilayer_leaves, search_emb, son_index):
        super().__init__(in_queue, out_queue)
        self.inner_vector = inner_vector
        self.root = root
        self.beam_size = beam_size
        self.inner_cnt = inner_cnt
        self.multilayer_leaves = multilayer_leaves
        self.search_emb = search_emb
        self.son_index = son_index

    def __call__(self, att, idx=None):
        # batch size should be 1, att should be an attention for only 1 token
        assert att.dim() == 1

        if self.beam_size > 1:
            return self.__beam_search(att)
        else:
            return self.__greedy_search_multilayer(att)

    def __greedy_search(self, att):
        prob, node = 1., self.root
        while node.tokenid is None:
            prob_left = torch.sigmoid(
                torch.dot(att, self.inner_vector[node.idx])).item()
            if prob_left >= 0.5:
                prob *= prob_left
                node = node.left
            else:
                prob *= 1. - prob_left
                node = node.right
        return [prob], [node.tokenid]

    def __greedy_search_multilayer(self, att):
        prob, node_idx = 1., 0
        att = att.unsqueeze(-1)
        while node_idx < self.inner_cnt:
            leaf_prob = torch.sigmoid(
                torch.matmul(self.search_emb[node_idx], att))
            leaf_prob = leaf_prob.view(self.multilayer_leaves, -1).prod(-1)
            leaf_prob, son_idx = leaf_prob.max(-1)
            prob = prob * leaf_prob
            node_idx = self.son_index[node_idx][son_idx]
        return [prob], [node_idx-self.inner_cnt]

    def __beam_search(self, att):
        class Candidate:
            def __init__(self, prob, node):
                self.prob = prob
                self.node = node

            def __lt__(self, other):
                return self.prob < other.prob
        candidates = []
        _candidates = []
        token_cnt = 0
        heapq.heappush(candidates, Candidate(1.0, self.root))

        while token_cnt < len(candidates):
            token_cnt = 0
            while len(candidates) > 0:
                cur = heapq.heappop(candidates)

                if cur.node.tokenid is not None:  # leaf node
                    token_cnt += 1
                    heapq.heappush(_candidates, cur)
                else:  # inner node
                    prob_left = torch.sigmoid(
                        torch.dot(att, self.inner_vector[cur.node.idx])).item()
                    prob_right = 1.0 - prob_left
                    prob_left *= cur.prob
                    prob_right *= cur.prob
                    heapq.heappush(_candidates, Candidate(
                        prob_left, cur.node.left))
                    heapq.heappush(_candidates, Candidate(
                        prob_right, cur.node.right))

            while len(_candidates) > self.beam_size:
                cur = heapq.heappop(_candidates)
                if cur.node.tokenid is not None:
                    token_cnt -= 1
            candidates = _candidates
            _candidates = []

        top_k_prob, top_k_index = [], []
        for cur in candidates:
            top_k_prob.append(cur.prob)
            top_k_index.append(cur.node.tokenid)

        return top_k_prob, top_k_index
