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
