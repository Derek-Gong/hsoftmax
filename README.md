# hsoftmax
Implementation of hierarchical softmax based on Huffman tree

## Usage
Run the script as follows:  
```
python wenet/utils/huffman_tree.py \
--train-data dir/to/format.data \
--dict dir/to/lang_char.txt \
--tree-save-dir dir/to/tree
```
Add the following lines to train and test yaml files
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
## Description
This project is based on [wenet](https://github.com/wenet-e2e/wenet).
Please reference that for more info about running an experiment.  
We implemented hierarchical softmax GPU/CPU training and decoding algorithm to leverage word frequence information for better performance on low-resources corpora and faster decoding speed compared to softmax.

We only add and modify several files based on wenet:  
wenet/utils/huffman_tree.py: For building a Huffman tree based on frequencies of train set  
wenet/transformer/hsoftmax_layer.py: For GPU training and GPU/CPU decoding  
wenet/utils/hsoftmax_processpool.py: Helper classes included for cpu parallel decoding in hsoftmax_layer.py  
wenet/transformer/asr_model.py: Several lines are added to enable hsoftmax  

## Performance
We reached a speedup in throughput of 4.19 in Librispeech using 10000 bpe tokens

## Implementation Details
![Implementation Details](implementation_details.jpg)
