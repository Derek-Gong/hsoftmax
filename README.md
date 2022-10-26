# Hierarchical Softmax (HSoftmax)
Implementation of hierarchical softmax based on Huffman tree

## Description
This project is based on [wenet](https://github.com/wenet-e2e/wenet).
Please reference that for more info about running an experiment.  
To run an experiment, you need to install the wenet first (commit 15ccd07), then replace the wenet fold (source files) with ours.  

We implemented hierarchical softmax GPU/CPU training and decoding algorithm to leverage word frequence information for better performance on low-resources corpora and faster decoding speed compared to softmax.

## Usage
1) Build Huffman tree on train set text  
2) Train model with hierarchical softmax  
3) Test model with hierarchical softmax  

## Usage Details
Run the script as follows to build Huffman tree:  
```
python wenet/utils/huffman_tree.py \
--train-data dir/to/format.data \
--dict dir/to/lang_char.txt \
--tree-save-dir dir/to/tree
```
Add the following lines to train and test yaml files to enable hsoftmax
```
hsoftmax:
    # the value of tree-save-dir given to huffman_tree.py
    huffman_tree_dir: "dir/to/tree"
    # number of CPU parallel workers for CPU decoding.
    num_workers: 40
    # tree search beam_size, we suggest 1 for greedy search, in most cases this is good enough.
    beam_size: 1 
    # the number of simultaneous inference layer for each iteration of hsoftmax tree search, 1 for layer by layer search. 
    # we suggest 2 or 3, while larger value may lead to performance degeneration.
    multilayer_decoding: 2 
```
Or you can specify hsoftmax decoding configs using command line arguments in recognize.py
```
--hsoftmax_beam_size 1
--hsoftmax_multilayer_decoding 2
```

## Source Code for the Paper
``` bibtex
@article{liu2022hsoftmax,
  title={Hierarchical Softmax for End-to-End Low-resource Multilingual Speech Recognition},
  author={Qianying Liu, Yuhang Yang, Zhuo Gong, Sheng Li, Chenchen Ding, Nobuaki Minematsu, Hao Huang, Fei Cheng, Sadao Kurohashi},
  journal={arXiv preprint arXiv:2204.03855},
  year={2022}
}
```

## Performance
We reached a speedup in throughput of 4.19 in Librispeech using 10000 bpe tokens
![result](docs/result.png)

## Core Concepts about HSoftmax GPU Training
![Implementation Details](docs/implementation_details.png)
