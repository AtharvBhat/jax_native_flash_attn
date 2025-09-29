from __future__ import annotations

import math
from typing import Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Sharding

from ._flash_attention import flash_attention as _flash_attention_core

Array = jax.Array


def _ensure_4d(t: Optional[Array]) -> Optional[Array]:
    if t is None:
        return None
    arr = jnp.asarray(t)
    if arr.ndim < 4:
        expand_axes = tuple(range(4 - arr.ndim))
        arr = jnp.expand_dims(arr, axis=expand_axes)
    return arr


def _broadcast_to(shape: Sequence[int], value: Array, dtype: jnp.dtype | None = None) -> Array:
    arr = jnp.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return jnp.broadcast_to(arr, shape)


def _sequence_mask(lengths: Optional[Array], target_len: int) -> Optional[Array]:
    if lengths is None:
        return None
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    positions = jnp.arange(target_len, dtype=jnp.int32)
    return positions < lengths[..., None]


def _causal_mask(t: int, s: int) -> Array:
    q_pos = jnp.arange(t, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(s, dtype=jnp.int32)[None, :]
    return q_pos >= k_pos


def _local_window_mask(t: int, s: int, window: Tuple[int, int]) -> Array:
    left, right = window
    q_pos = jnp.arange(t, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(s, dtype=jnp.int32)[None, :]
    return (k_pos >= (q_pos - left)) & (k_pos <= (q_pos + right))


def flash_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    *,
    scale: Optional[float] = None,
    is_causal: bool = False,
    query_seq_lengths: Optional[Array] = None,
    key_value_seq_lengths: Optional[Array] = None,
    local_window_size: Optional[int | Tuple[int, int]] = None,
    implementation: Literal["xla", "cudnn", None] = None,
    dropout_rate: float = 0.0,
    rng: Optional[jax.random.PRNGKey],
    blocksize_q: Optional[int] = None,
    blocksize_k: Optional[int] = None,
    dtype: Optional[jnp.dtype] = None,
    precision: Optional[lax.PrecisionLike] = None,
    output_sharding: Optional[Sharding] = None,
) -> Array:
    """Compute attention using a pure-JAX FlashAttention kernel."""
    if dropout_rate < 0.0 or dropout_rate > 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if rng is None:
            raise ValueError("An rng key must be provided to flash_attention")
    if local_window_size is not None and isinstance(local_window_size, int):
        local_window_size = (local_window_size, local_window_size)
    if local_window_size is not None and (
        local_window_size[0] < 0 or local_window_size[1] < 0
    ):
        raise ValueError("local_window_size entries must be non-negative")
    if implementation not in (None, "xla"):
        raise NotImplementedError(
            "Only the pure JAX implementation is available; implementation must be None or 'xla'."
        )

    output_shape = jnp.asarray(query).shape

    query_sharding = getattr(query, "sharding", None)
    if output_sharding is None:
        output_sharding = query_sharding
    elif query_sharding is not None and output_sharding != query_sharding:
        raise AssertionError("output_sharding must match the query sharding")

    query_arr = _ensure_4d(query)
    key_arr = _ensure_4d(key)
    value_arr = _ensure_4d(value)
    bias_arr = _ensure_4d(bias)
    mask_arr = _ensure_4d(mask)

    B, S, K, H = key_arr.shape
    T = query_arr.shape[1]
    N = query_arr.shape[2]
    if value_arr.shape != key_arr.shape:
        raise ValueError("value must have the same shape as key")
    if query_arr.shape[0] != B:
        raise ValueError("query and key must have the same batch dimension")
    if query_arr.shape[-1] != H:
        raise ValueError("query and key must have matching head dimensions")
    if N != K:
        raise ValueError(
            "The number of query heads must match the number of key/value heads for FlashAttention"
        )

    q_lengths = _sequence_mask(query_seq_lengths, T)
    kv_lengths = _sequence_mask(key_value_seq_lengths, S)

    mask_components = []
    if mask_arr is not None:
        mask_components.append(_broadcast_to((B, N, T, S), mask_arr, dtype=bool))
    if q_lengths is not None or kv_lengths is not None:
        if q_lengths is None:
            q_lengths = jnp.ones((B, T), dtype=bool)
        if kv_lengths is None:
            kv_lengths = jnp.ones((B, S), dtype=bool)
        seq_mask = (
            q_lengths[:, None, :, None] & kv_lengths[:, None, None, :]
        )
        mask_components.append(seq_mask)
    if is_causal:
        mask_components.append(_causal_mask(T, S)[None, None, :, :])
    if local_window_size is not None:
        mask_components.append(
            _local_window_mask(T, S, local_window_size)[None, None, :, :]
        )

    combined_mask: Optional[Array] = None
    for component in mask_components:
        combined_mask = component if combined_mask is None else (combined_mask & component)

    bias_broadcast = None
    if bias_arr is not None:
        bias_broadcast = _broadcast_to((B, N, T, S), bias_arr, dtype=query_arr.dtype)

    scale_val = (1.0 / math.sqrt(H)) if scale is None else scale
    q_scaled = query_arr * jnp.asarray(scale_val, dtype=query_arr.dtype)

    q_canonical = jnp.transpose(q_scaled, (0, 2, 1, 3))
    k_canonical = jnp.transpose(key_arr, (0, 2, 1, 3))
    v_canonical = jnp.transpose(value_arr, (0, 2, 1, 3))
    mask_canonical = None if combined_mask is None else combined_mask.astype(bool)
    bias_canonical = None if bias_broadcast is None else bias_broadcast

    attn_out = _flash_attention_core(
        q_canonical,
        k_canonical,
        v_canonical,
        mask_canonical,
        bias_canonical,
        dropout=dropout_rate,
        rng=rng,
        blocksize_q=blocksize_q,
        blocksize_k=blocksize_k,
        dtype=dtype,
        precision=precision,
        output_sharding=output_sharding,
    )

    attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
    attn_out = jnp.reshape(attn_out, output_shape)
    if dtype is not None:
        attn_out = attn_out.astype(dtype)
    return attn_out


__all__ = ["flash_attention"]
