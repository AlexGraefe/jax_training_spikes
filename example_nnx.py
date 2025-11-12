import gc
from time import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tqdm
import matplotlib.pyplot as plt

import numpy as np

class VisionTransformer(nnx.Module):
    """ Implements the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        num_classes (int): Number of classes in the classification. Defaults to 1000.
        in_channels (int): Number of input channels in the image (such as 3 for RGB). Defaults to 3.
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Size of the patches extracted from the image. Defaults to 16.
        num_layers (int): Number of transformer encoder layers. Defaults to 12.
        num_heads (int): Number of attention heads in each transformer layer. Defaults to 12.
        mlp_dim (int): Dimension of the hidden layers in the feed-forward/MLP block. Defaults to 3072.
        hidden_size (int): Dimensionality of the embedding vectors. Defaults to 3072.
        dropout_rate (int): Dropout rate (for regularization). Defaults to 0.1.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.

    """
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        img_size: int = 32,
        patch_size: int = 2,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 256,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # Calculate the number of patches generated from the image.
        n_patches = (img_size // patch_size) ** 2
        # Patch embeddings:
        # - Extracts patches from the input image and maps them to embedding vectors
        #   using `flax.nnx.Conv` (convolutional layer).
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

        # Positional embeddings (add information about image patch positions):
        # Set the truncated normal initializer (using `jax.nn.initializers.truncated_normal`).
        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        # The learnable parameter for positional embeddings (using `flax.nnx.Param`).
        self.position_embeddings = nnx.Param(
            initializer(rngs.params(), (1, n_patches + 1, hidden_size), jnp.float32)
        ) # Shape `(1, n_patches +1, hidden_size`)
        # The dropout layer.
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # CLS token (a special token prepended to the sequence of patch embeddings)
        # using `flax.nnx.Param`.
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))

        # Transformer encoder (a sequence of encoder blocks for feature extraction).
        # - Create multiple Transformer encoder blocks (with `nnx.Sequential`
        # and `TransformerEncoder(nnx.Module)` which is defined later).
        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs)
            for i in range(num_layers)
        ])
        # Layer normalization with `flax.nnx.LayerNorm`.
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)

        # Classification head (maps the transformer encoder to class probabilities).
        self.classifier = nnx.Linear(hidden_size, num_classes, rngs=rngs)

    # The forward pass in the ViT model.
    def __call__(self, x: jax.Array) -> jax.Array:
        # Image patch embeddings.
        # Extract image patches and embed them.
        patches = self.patch_embeddings(x)
        # Get the batch size of image patches.
        batch_size = patches.shape[0]
        # Reshape the image patches.
        patches = patches.reshape(batch_size, -1, patches.shape[-1])

        # Replicate the CLS token for each image with `jax.numpy.tile`
        # by constructing an array by repeating `cls_token` along `[batch_size, 1, 1]` dimensions.
        cls_token = jnp.tile(self.cls_token, [batch_size, 1, 1])
        # Concatenate the CLS token and image patch embeddings.
        x = jnp.concat([cls_token, patches], axis=1)
        # Create embedded patches by adding positional embeddings to the concatenated CLS token and image patch embeddings.
        embeddings = x + self.position_embeddings
        # Apply the dropout layer to embedded patches.
        embeddings = self.dropout(embeddings)

        # Transformer encoder blocks.
        # Process the embedded patches through the transformer encoder layers.
        x = self.encoder(embeddings)
        # Apply layer normalization
        x = self.final_norm(x)

        # Extract the CLS token (first token), which represents the overall image embedding.
        x = x[:, 0]

        # Predict class probabilities based on the CLS token embedding.
        return self.classifier(x)


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block in the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        hidden_size (int): Input/output embedding dimensionality.
        mlp_dim (int): Dimension of the feed-forward/MLP block hidden layer.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.
    """
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        # First layer normalization using `flax.nnx.LayerNorm`
        # before we apply Multi-Head Attentn.
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        # The Multi-Head Attention layer (using `flax.nnx.MultiHeadAttention`).
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
        )
        # Second layer normalization using `flax.nnx.LayerNorm`.
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        # The MLP for point-wise feedforward (using `flax.nnx.Sequential`, `flax.nnx.Linear, flax.nnx.Dropout`)
        # with the GeLU activation function (`flax.nnx.gelu`).
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    # The forward pass through the transformer encoder block.
    def __call__(self, x: jax.Array) -> jax.Array:
        # The Multi-Head Attention layer with layer normalization.
        x = x + self.attn(self.norm1(x))
        # The feed-forward network with layer normalization.
        x = x + self.mlp(self.norm2(x))
        return x

model = VisionTransformer(num_classes=10)

seed = 12
train_batch_size = 64
val_batch_size = 2 * train_batch_size

num_epochs = 3
learning_rate = 0.001
momentum = 0.8
total_steps = 1000

lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)
iterate_subsample = np.linspace(0, num_epochs * total_steps, 100)

optimizer = nnx.ModelAndOptimizer(model, optax.sgd(lr_schedule, momentum, nesterov=True))

def compute_losses_and_logits(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits

@nnx.jit
def train_step(
    model: nnx.Module, batch: dict[str, np.ndarray]
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    # optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, np.ndarray], eval_metrics: nnx.MultiMetric
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)
    loss, logits = compute_losses_and_logits(model, images, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )

eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)


train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {
    "val_loss": [],
    "val_accuracy": [],
}

import tqdm


bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


def train_one_epoch():
    model.train()  # Set model to the training mode: e.g. update batch statistics
    batch = {
        "image": jnp.zeros((train_batch_size, 32, 32, 3), dtype=jnp.float32),
        "label": jnp.ones((train_batch_size,), dtype=jnp.int32) / 10,
    }
    
    losses = []
    training_times_gc_enabled = []
    for s in tqdm.tqdm(range(1000)):
        start_time = time()
        loss = train_step(model, batch)
        losses.append(loss)
        if s > 0:
            training_times_gc_enabled.append(time() - start_time)

    gc.disable()
    training_times_gc_disabled = []
    for s in tqdm.tqdm(range(1000)):
        start_time = time()
        loss = train_step(model, batch)
        losses.append(loss)
        if s > 0:
            training_times_gc_disabled.append(time() - start_time)

    plt.figure(figsize=(8, 4))
    plt.plot(training_times_gc_enabled, marker="o", label="GC Enabled")
    plt.plot(training_times_gc_disabled, marker="x", label="GC Disabled")
    plt.xlabel("Batch Count")
    plt.ylabel("Training Time (s)")
    plt.title("NNX Batch Training Times")
    plt.legend()
    plt.grid(True)
    plt.show()

train_one_epoch()