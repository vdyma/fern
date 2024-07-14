# Fern: The Multimodal Magic 🌿

_In the realm where technology and nature intertwine, Fern emerges as a beacon of innovation and mystery. This deep learning model transcends the ordinary, seamlessly blending text, vision, and audio into a harmonious symphony of understanding. Much like its mythological namesake, the fern—believed to possess hidden powers and secrets of the forest—Fern delves into the depths of multimodal data, uncovering patterns and insights that elude conventional models. Enigmatic and almost magical, Fern transforms the way we perceive and interact with information, revealing the unseen and unheard with an elegance that borders on the supernatural._

## What is Fern?

Fern is an ongoing effort to create a model similar to [Chameleon](https://arxiv.org/abs/2405.09818), but which would cover all modalities. Currently it resembles [Llama 2](https://arxiv.org/abs/2307.09288) in terms of architecture.

## Components

The following system components are implemented:

- [Transformer](https://arxiv.org/abs/1706.03762) architecture
    - [RMSNorm](https://arxiv.org/abs/1910.07467) normalization
    - [RoPE](https://arxiv.org/abs/2104.09864) positional embeddings
    - [SwiGLU](https://arxiv.org/abs/2002.05202) activation function
- [Byte-Pair Encoding](https://arxiv.org/abs/1508.07909) Tokenizer

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
