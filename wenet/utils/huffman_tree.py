import os
import argparse
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

    def __init__(self, freq, token):
        self.freq = freq
        self.token = token
        self.idx = None
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

    def build_idx(self, token_id, nxt = 0):
        if self.token is not None:
            self.idx = token_id[self.token]
            return nxt

        self.idx = nxt
        nxt += 1
        if self.left is not None:
            nxt = self.left.build_idx(token_id, nxt)
        if self.right is not None:
            nxt = self.right.build_idx(token_id, nxt)
        return nxt

    def toJSON(self):
        return json.dumps(self, cls=Node.NodeEncoder, ensure_ascii=False, indent=1)

    @staticmethod
    def fromJSON(json_string):
        def as_Node(dct):
            if isinstance(dct, Node):
                return dct
            node = Node(dct['freq'], dct['token'])
            node.idx = dct['idx']
            if dct['left'] is not None:
                node.left = as_Node(dct['left'])
            if dct['right'] is not None:
                node.right = as_Node(dct['right'])
            return node       

        return json.loads(json_string, object_hook=as_Node)
            
class HuffmanTree:
    def __init__(self, counter, token_id):
        self.token_id = token_id
        self.root = None
        self.path_mask_sign = None
        self.path_mask_bias = None
        self.leaf_cnt = 0
        self.inner_cnt = 0
        self.info_dict = None

        if counter is not None:
            self.__build_tree(counter)
            if token_id is not None:
                self.__build_path_mask(token_id)

    def __build_tree(self, counter):
        q = PriorityQueue()
        tot_cnt = sum(counter.values())
        for token, cnt in counter.items():
            freq = cnt / tot_cnt
            node = Node(freq, token)
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
            with codecs.open(path_name,"r", encoding='utf-8') as jsonfile:
                return jsonfile.read()
        def loadnp(filename):
            path_name = os.path.join(save_dir, filename)
            with open(path_name,"rb") as f:
                return np.load(f)
        tree = HuffmanTree(None, None)
        # load struct from json
        tree.root = Node.fromJSON(load_string('tree_struct'))
        # load path_mask_sign from npy
        tree.path_mask_sign = loadnp('path_mask_sign.npy')
        # load path_mask_bias from npy
        tree.path_mask_bias = loadnp('path_mask_bias.npy')
        # load info_dict from json
        tree.info_dict = json.loads(load_string('tree_info'))
        tree.leaf_cnt = tree.info_dict['leaf_cnt']
        tree.inner_cnt = tree.leaf_cnt - 1
        
        return tree

    def save(self, save_dir):
        def save2js(filename, js):
            path_name = os.path.join(save_dir, filename)
            with codecs.open(path_name,"w", encoding='utf-8') as jsonfile:
                jsonfile.write(js)
        def save2np(filename, data):
            path_name = os.path.join(save_dir, filename)
            with open(path_name,"wb") as f:
                np.save(f, data)
        # save struct in json
        save2js('tree_struct', self.toJSON())
        # save path_mask_sign in npy
        save2np('path_mask_sign.npy', self.path_mask_sign)
        # save path_mask_bias in npy
        save2np('path_mask_bias.npy', self.path_mask_bias)
        # save info in string
        save2js('tree_info',  json.dumps(self.info()))

    def toJSON(self):
        return self.root.toJSON()

    def __build_path_mask(self, token_id):
        self.root.build_idx(token_id)
        self.info()
        self.leaf_cnt = self.info_dict['leaf_cnt']
        self.inner_cnt = self.leaf_cnt - 1
        
        self.path_mask_sign = np.zeros((self.leaf_cnt, self.inner_cnt), dtype=int)
        self.path_mask_bias = np.ones((self.leaf_cnt, self.inner_cnt), dtype=int)

        def dfs(node, 
            sign = np.zeros((self.inner_cnt,), dtype=int), 
            bias = np.ones((self.inner_cnt,), dtype=int)):
            # node is leaf
            if node.left is None and node.right is None:
                self.path_mask_sign[token_id[node.token]] = sign
                self.path_mask_bias[token_id[node.token]] = bias
            # has child
            if node.left is not None:
                sign[node.idx] = 1
                bias[node.idx] = 0
                dfs(node.left, sign, bias)
                sign[node.idx] = 0
                bias[node.idx] = 1
            if node.right is not None:
                sign[node.idx] = -1
                dfs(node.right, sign, bias)
                sign[node.idx] = 0
        
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
            self.info_dict = {'depth': depth(self.root), 'leaf_cnt': cnt_leaf(self.root)}
        return self.info_dict

def build_huffman_tree(data_file, dict_file, save_dir):
    token_counter = Counter()
    line_cnt = 0
    with codecs.open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_cnt = line_cnt + 1
            arr = line.strip().split('\t')

            token_seq=arr[4].split(':')[1]
            
            tokens=token_seq.split( )
            
            for token in tokens:
                token_counter[token] += 1
    
    token_counter["<sos/eos>"] = line_cnt

    token_id = {} 
    with codecs.open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split()
            token_id[token]=int(idx)
            if token not in token_counter:
                token_counter[token] = 0
    
    tree = HuffmanTree(token_counter, token_id)
    tree.save(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build huffman tree on train set')
    parser.add_argument('--train-data', required=True, help='train data file')
    parser.add_argument('--dict', required=True, help='token to id mapping')
    parser.add_argument('--tree-save-dir', required=True, help='token to id mapping')
    args = parser.parse_args()
    build_huffman_tree(args.train_data, args.dict, args.tree_save_dir)