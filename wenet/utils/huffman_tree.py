import os
from pathlib import Path
import argparse
import logging
import json
import codecs
import numpy as np
from collections import Counter
from queue import PriorityQueue


class Node:
    class NodeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Node):
                return obj.__dict__
            else:
                return json.JSONEncoder.default(self, obj)

    def __init__(self, freq, tokenid):
        self.freq = freq
        self.tokenid = tokenid
        self.idx = None
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def build_idx(self, nxt=0):
        if self.tokenid is not None:
            self.idx = self.tokenid
            return nxt

        self.idx = nxt
        nxt += 1
        if self.left is not None:
            nxt = self.left.build_idx(nxt)
        if self.right is not None:
            nxt = self.right.build_idx(nxt)
        return nxt

    def toJSON(self):
        return json.dumps(self, cls=Node.NodeEncoder, ensure_ascii=False, indent=1)

    @staticmethod
    def fromJSON(json_string):
        def as_Node(dct):
            if isinstance(dct, Node):
                return dct
            node = Node(dct['freq'], dct['tokenid'])
            node.idx = dct['idx']
            if dct['left'] is not None:
                node.left = as_Node(dct['left'])
            if dct['right'] is not None:
                node.right = as_Node(dct['right'])
            return node

        return json.loads(json_string, object_hook=as_Node)


class HuffmanTree:
    def __init__(self, counter):
        self.root = None
        self.path_mask_sign = None
        self.path_mask_bias = None
        self.path_index = None
        self.path_sign = None
        self.path_bias = None
        self.leaf_cnt = 0
        self.inner_cnt = 0
        self.depth = 0
        self.info_dict = None

        if counter is not None:
            self.__build_tree(counter)
            self.__build_tree_index()
            self.__build_path_index()

    def __build_tree(self, counter):
        q = PriorityQueue()
        tot_cnt = sum(counter.values())
        for tokenid, cnt in counter.items():
            freq = cnt / tot_cnt
            node = Node(freq, tokenid)
            q.put(node)

        while q.qsize() > 1:
            left, right = q.get(), q.get()
            parent = Node(left.freq + right.freq, None)
            parent.left, parent.right = left, right
            q.put(parent)

        self.root = q.get()

    @staticmethod
    def load(save_dir):
        def load_string(filename):
            path_name = os.path.join(save_dir, filename)
            with codecs.open(path_name, "r", encoding='utf-8') as jsonfile:
                return jsonfile.read()

        def loadnp(filename):
            path_name = os.path.join(save_dir, filename)
            with open(path_name, "rb") as f:
                return np.load(f)
        tree = HuffmanTree(None)
        # load struct from json
        tree.root = Node.fromJSON(load_string('tree_struct'))
        # load path_index from npy
        tree.path_index = loadnp('path_index.npy')
        # load path_sign from npy
        tree.path_sign = loadnp('path_sign.npy')
        # load path_bias from npy
        tree.path_bias = loadnp('path_bias.npy')
        # load info_dict from json
        tree.info_dict = json.loads(load_string('tree_info'))
        tree.leaf_cnt = tree.info_dict['leaf_cnt']
        tree.inner_cnt = tree.leaf_cnt - 1
        tree.depth = tree.info_dict['depth']

        return tree

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        def save2js(filename, js):
            path_name = os.path.join(save_dir, filename)
            with codecs.open(path_name, "w", encoding='utf-8') as jsonfile:
                jsonfile.write(js)

        def save2np(filename, data):
            path_name = os.path.join(save_dir, filename)
            with open(path_name, "wb") as f:
                np.save(f, data)
        # save struct in json
        save2js('tree_struct', self.toJSON())
        # save info in string
        save2js('tree_info', json.dumps(self.info()))
        # save path_index in npy
        save2np('path_index.npy', self.path_index)
        # save path_sign in npy
        save2np('path_sign.npy', self.path_sign)
        # save path_bias in npy
        save2np('path_bias.npy', self.path_bias)

    def toJSON(self):
        return self.root.toJSON()

    def __build_tree_index(self):
        self.root.build_idx()
        self.info()
        self.leaf_cnt = self.info_dict['leaf_cnt']
        self.inner_cnt = self.leaf_cnt - 1
        self.depth = self.info_dict['depth']

    def __build_path_index(self):
        self.path_index = np.zeros(
            (self.leaf_cnt, self.depth), dtype=int)
        self.path_sign = np.zeros(
            (self.leaf_cnt, self.depth), dtype=np.int8)
        self.path_bias = np.ones(
            (self.leaf_cnt, self.depth), dtype=np.int8)

        def dfs(node, depth=0,
                index=np.zeros((self.depth,), dtype=int),
                sign=np.zeros((self.depth,), dtype=np.int8),
                bias=np.ones((self.depth,), dtype=np.int8)):
            # node is leaf
            if node.left is None and node.right is None:
                self.path_index[node.idx] = index
                self.path_sign[node.idx] = sign
                self.path_bias[node.idx] = bias
                return
            # has child
            index[depth] = node.idx
            if node.left is not None:
                sign[depth] = 1
                bias[depth] = 0
                dfs(node.left, depth+1, index, sign, bias)
                sign[depth] = 0
                bias[depth] = 1
            if node.right is not None:
                sign[depth] = -1
                dfs(node.right, depth+1, index, sign, bias)
                sign[depth] = 0
            index[depth] = 0

        dfs(self.root)

    def info(self):
        def depth(node):
            if node is None:
                return 0
            left, right = 0, 0
            if node.left is not None:
                left = depth(node.left)
            if node.right is not None:
                right = depth(node.right)
            return 1 + max(left, right)

        def cnt_leaf(node):
            if node is None:
                return 0
            left, right = 0, 0
            if node.left is not None:
                left = cnt_leaf(node.left)
            if node.right is not None:
                right = cnt_leaf(node.right)
            return (left + right) or 1

        if self.info_dict is None:
            self.info_dict = {'depth': depth(
                self.root), 'leaf_cnt': cnt_leaf(self.root)}
        return self.info_dict


def build_huffman_tree(data_file, dict_file, save_dir):
    tokenid_counter = Counter()
    line_cnt = 0
    with codecs.open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                arr = line.strip().split('\t')

                idx_seq = arr[5].split(':')[1]

                idxs = idx_seq.split()

                line_cnt = line_cnt + 1

                for idx in idxs:
                    tokenid_counter[int(idx)] += 1
            except:
                logging.warning('Omit an ill-formed line: ' + line.strip())

    token2id = {}
    with codecs.open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split()
            idx = int(idx)
            token2id[token] = idx
            if idx not in tokenid_counter:
                tokenid_counter[idx] = 0
    tokenid_counter[token2id["<sos/eos>"]] = line_cnt

    tree = HuffmanTree(tokenid_counter)
    tree.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='build huffman tree on train set')
    parser.add_argument('--train-data', required=True,
                        help='train data file')
    parser.add_argument('--dict', required=True,
                        help='token to id mapping')
    parser.add_argument('--tree-save-dir', required=True,
                        help='tree saving directory')
    args = parser.parse_args()
    build_huffman_tree(args.train_data, args.dict, args.tree_save_dir)
