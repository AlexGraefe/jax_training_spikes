import copy
import functools
import time

import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax  # https://github.com/deepmind/optax

from jaxtyping import Array, Float, PRNGKeyArray

import tqdm
import gc

# Hyperparameters
lr = 0.0001
dropout_rate = 0.1
beta1 = 0.9
beta2 = 0.999
batch_size = 64
patch_size = 2
num_steps = 2000
image_size = (32, 32, 3)
num_patches = (image_size[0]//patch_size)**2
embedding_dim = 512
hidden_dim = 256
num_heads = 8
num_layers = 12
height, width, channels = image_size
num_classes = 10


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Embedding
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(
        self, x
    ):
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)

        return x
    
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ):
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x)

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, inference=not enable_dropout, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout, key=key2)

        x = x + input_x

        return x
    
class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.Sequential
    num_layers: int

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
    ):
        key1, key2, key3, key4, key5 = jr.split(key, 5)

        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)

        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))

        self.cls_token = jr.normal(key3, (1, embedding_dim))

        self.num_layers = num_layers

        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(self.num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=key5),
            ]
        )

    def __call__(
        self,
        x,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ):
        x = self.patch_embedding(x)

        x = jnp.concatenate((self.cls_token, x), axis=0)

        x += self.positional_embedding[
            : x.shape[0]
        ]  # Slice to the same length as x, as the positional embedding may be longer.

        dropout_key, *attention_keys = jr.split(key, num=self.num_layers + 1)

        x = self.dropout(x, inference=not enable_dropout, key=dropout_key)

        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, enable_dropout, key=attention_key)

        x = x[0]  # Select the CLS token.
        x = self.mlp(x)

        return x

def flatten_twice(model):
    unflattened = jax.tree.flatten(model)
    unflattened = jax.tree.flatten(model)

def flatten_once(model):
    unflattened = jax.tree.flatten(model)

def train(
    model: VisionTransformer,
    num_steps: int,
    print_every: int = 200,
    key=None,
):
    # Benchmark flatten_once with GC enabled
    gc.enable()
    flatten_once_gc_enabled = []
    for step in tqdm.tqdm(range(num_steps), desc="flatten_once (GC enabled)"):
        start_time = time.time()
        flatten_once(model)
        if step > 0:  # exclude first run for JIT compilation
            flatten_once_gc_enabled.append(time.time() - start_time)
    
    # Benchmark flatten_twice with GC enabled
    flatten_twice_gc_enabled = []
    for step in tqdm.tqdm(range(num_steps), desc="flatten_twice (GC enabled)"):
        start_time = time.time()
        flatten_twice(model)
        if step > 0:
            flatten_twice_gc_enabled.append(time.time() - start_time)
    
    # Benchmark flatten_once with GC disabled
    gc.disable()
    flatten_once_gc_disabled = []
    for step in range(num_steps):
        start_time = time.time()
        flatten_once(model)
        if step > 0:
            flatten_once_gc_disabled.append(time.time() - start_time)
    
    # Benchmark flatten_twice with GC disabled
    flatten_twice_gc_disabled = []
    for step in range(num_steps):
        start_time = time.time()
        flatten_twice(model)
        if step > 0:
            flatten_twice_gc_disabled.append(time.time() - start_time)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate global y-axis limits
    all_times = (flatten_once_gc_enabled + flatten_once_gc_disabled + 
                flatten_twice_gc_enabled + flatten_twice_gc_disabled)
    y_min, y_max = min(all_times), max(all_times)
    y_range = y_max - y_min
    y_min -= y_range * 0.05  # Add 5% padding
    y_max += y_range * 0.05
    
    # Plot flatten_once times
    ax1.plot(flatten_once_gc_enabled, marker="o", label="GC Enabled")
    ax1.plot(flatten_once_gc_disabled, marker="x", label="GC Disabled")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("flatten_once Benchmark")
    ax1.set_ylim(y_min, y_max)
    ax1.legend()
    ax1.grid(True)
    
    # Plot flatten_twice times
    ax2.plot(flatten_twice_gc_enabled, marker="o", label="GC Enabled")
    ax2.plot(flatten_twice_gc_disabled, marker="x", label="GC Disabled")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("flatten_twice Benchmark")
    ax2.set_ylim(y_min, y_max)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    return None, None

key = jr.PRNGKey(2003)

model = VisionTransformer(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    patch_size=patch_size,
    num_patches=num_patches,
    num_classes=num_classes,
    key=key,
)

model, state = train(model, num_steps, key=key)