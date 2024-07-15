# Fern: The Multimodal Magic 🌿

_In the realm where technology and nature intertwine, Fern emerges as a beacon of innovation and mystery. This deep learning model transcends the ordinary, seamlessly blending text, vision, and audio into a harmonious symphony of understanding. Much like its mythological namesake, the fern—believed to possess hidden powers and secrets of the forest—Fern delves into the depths of multimodal data, uncovering patterns and insights that elude conventional models. Enigmatic and almost magical, Fern transforms the way we perceive and interact with information, revealing the unseen and unheard with an elegance that borders on the supernatural._

## What is Fern?

Fern is an ongoing effort to create a model similar to [Chameleon](https://arxiv.org/abs/2405.09818), but which would cover all modalities. Currently it resembles [Llama 2](https://arxiv.org/abs/2307.09288) in terms of architecture.

## Overview

### Architecture

The following system components are implemented:

- [Transformer](https://arxiv.org/abs/1706.03762) architecture
    - [RMSNorm](https://arxiv.org/abs/1910.07467) normalization
    - [RoPE](https://arxiv.org/abs/2104.09864) positional embeddings
    - [SwiGLU](https://arxiv.org/abs/2002.05202) activation function
    - [Embedding sharing](https://arxiv.org/abs/2205.01068)
- [Byte-Pair Encoding](https://arxiv.org/abs/1508.07909) tokenizer

### Data and tokenization

Train data is all the books from [The Elder Scrolls](https://elderscrolls.com) video game series taken from [The Imperial Library](https://www.imperial-library.info/). BPE tokenizer is trained on the whole dataset to create 2048 new tokens, including the special `<|endoftext|>` token. This results in vocabulary containing 2304 tokens in total (256 byte tokens, 2047 pair tokens and 1 special token). Vocabulary of this size captures lore terms such as ` Valenwood` and ` Bosmer` as well as literature categories such as ` Fiction` and ` Narrative` in a single token which is learned nearly at the end of the tokenizer training. Further investigation on the optimal vocabulary size needed.

### Results

The model `dim` is 128, it has 32 layers with 8 heads each. Model is trained with the context size 512 tokens and batch size containing 32 examples. The model achieves 2.74 train loss and 3.52 validation loss after 10000 training steps. The training process took nearly 50 minutes on a single laptop Nvidia RTX 4060 GPU. 

## Install and run

1. Install [miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

2. Clone this repository. To clone the repository using [git](https://git-scm.com/), run:

```bash
git clone https://github.com/vdyma/fern
```

3. Open repository root directory as a working directory in your shell and create a new conda environment:

```bash
conda install --file environment.yaml
```

4. Activate conda environment:

```bash
conda activate fern
```

You are now ready to use the model. 

(Optional) In order to run Jupyter Notebooks, you need to additionally install [Jupyter Lab](https://jupyter.org/install) and `ipywidgets`:

```bash
conda install jupyterlab ipywidgets
```

You can then run jupyter as follows:

```bash
jupyter lab
```
