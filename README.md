# tinyGPT
Implementation of a decoder-only GPT model as taught by Andrej Karpathy for karpathy/nanoGPT

## Models
### GPT Model
[gpt.py](gpt.py): This includes the implementation for the character-level, decoder-only transformer that is able to generate text on prompt. 
1. `Head` class: implements a single self-attention head that holds its key, query, and value.
    - Key: what information this head contains.
    - Query: the information this head is looking (querying) for.
    - Value: the information this head provides, aggregrated using input.
2. `MultiHeads` class: implements n heads for multiple attentions in parallel.
3. `FeedForward` class: implements a feed forward for computation on a token level of information after communication within the transformer block.
4. `Block` class: implements the block that encapsulates the multi-attention and communication components, and is to be run n times by the network.

### Bigram Language Model
[bigram.py](bigram.py): This is a simple model that outlines the basics of text generation using a character level language model. It includes just a single Embedding layer used for generation. 

### Notes
This was built following the [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY) by Andrej Karpathy. The model is built using architecture and implementation details outlined in [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. To introduce ability to generate meaningful reply related to prompt, the encoder and cross attention component from the paper would need to be added to this model.