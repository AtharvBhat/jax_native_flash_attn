# jax_native_flash_attn

`jax_native_flash_attn` provides a drop-in replacement for
`jax.nn.dot_product_attention` implemented entirely in JAX. It repackages the
pure JAX FlashAttention kernel from Erfan Zare's
[`jax-flash-attn2`](https://github.com/erfanzar/jax-flash-attn2) repository into
a lighter-weight, more portable library that keeps the memory footprint of
FlashAttention while supporting the same masking, biasing, and dropout options
exposed by the upstream API.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency
management. To create a virtual environment with the runtime and development
dependencies, run:

```sh
uv sync
```

## Usage

```python
import jax
import jax.numpy as jnp

from jax_native_flash_attn import flash_attention

query = jnp.ones((2, 128, 8, 64))
key = jnp.ones((2, 128, 8, 64))
value = jnp.ones((2, 128, 8, 64))

output = flash_attention(query, key, value, is_causal=True)
```

All arguments supported by `jax.nn.dot_product_attention`—including bias,
Boolean masks, sequence lengths, causal masking, and local window masking—are
available. The kernel can be jitted and differentiated in the same way as the
reference implementation.

## Testing

The test-suite exercises forward values, gradients, VJPs, dtype coverage, and
dropout parity against the `jax.nn.dot_product_attention` reference. Run it
locally with:

```sh
uv run pytest
```

## Benchmarking

To compare the native FlashAttention kernel to the XLA-based implementation of
`jax.nn.dot_product_attention`, run the benchmark script:

```sh
uv run python benchmarks/benchmark_flash_attention.py
```

By default this executes a few representative batch/head/sequence configurations
and reports per-call latency for both implementations.

## Credits

This project is a portability-focused repackaging of the pure JAX kernel
originally implemented by Erfan Zare in
[`erfanzar/jax-flash-attn2`](https://github.com/erfanzar/jax-flash-attn2).
All credit for the FlashAttention implementation belongs to Erfan; this library
simply trims dependencies and redistributes the code for broader use.
