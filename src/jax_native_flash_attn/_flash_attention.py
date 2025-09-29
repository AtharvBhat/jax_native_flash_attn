from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec, SingleDeviceSharding

Array = jax.Array


def _neg_inf(dtype: jnp.dtype) -> Array:
    """Returns a large negative value that is safe for masking."""
    info = jnp.finfo(jnp.dtype(dtype))
    return jnp.array(info.min, dtype=dtype)


def _apply_sharding(
    arr: Array, sharding: Optional[jax.sharding.Sharding], *, rank: int
) -> Array:
    """Materialise ``arr`` with ``sharding`` if one was requested.

    The FlashAttention kernel uses the query sharding as the reference layout for
    intermediates with a lower rank (for example, the log-sum-exp buffer which
    drops the feature dimension).  ``rank`` captures the dimensionality of the
    target array so we can project a higher-rank ``PartitionSpec`` down to the
    matching prefix before placing the array on devices.
    """

    if sharding is None:
        return arr

    if isinstance(sharding, NamedSharding):
        spec = tuple(sharding.spec)
        if len(spec) < rank:
            spec = spec + (None,) * (rank - len(spec))
        elif len(spec) > rank:
            spec = spec[:rank]
        target_spec = PartitionSpec(*spec)
        target_sharding = NamedSharding(sharding.mesh, target_spec)
        return jax.device_put(arr, target_sharding)

    if isinstance(sharding, SingleDeviceSharding):
        device = getattr(sharding, "device", None)
        if device is None:
            device = getattr(sharding, "_device", None)
        if device is not None:
            return jax.device_put(arr, device)
        return arr

    return arr


def _compute_block_sizes(q_len: int, k_len: int, blocksize_q: Optional[int], blocksize_k: Optional[int]) -> Tuple[int, int]:
    """Choose block sizes that evenly divide the sequence lengths."""
    bs_q = min(blocksize_q or q_len, q_len)
    bs_k = min(blocksize_k or k_len, k_len)
    bs_q = math.gcd(bs_q, q_len) or q_len
    bs_k = math.gcd(bs_k, k_len) or k_len
    return bs_q, bs_k


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(5, 6, 8, 9, 10, 11),
)
def _flash_attention_core(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array],
    bias: Optional[Array],
    output_sharding: Optional[jax.sharding.Sharding],
    dropout: float,
    rng: jax.random.PRNGKey,
    blocksize_q: int,
    blocksize_k: int,
    dtype: Optional[jnp.dtype],
    precision: Optional[lax.PrecisionLike],
) -> Array:
    output, _ = _fwd_flash_attention(
        query,
        key,
        value,
        mask,
        bias,
        output_sharding,
        dropout,
        rng,
        blocksize_q,
        blocksize_k,
        dtype,
        precision,
    )
    return output


def _fwd_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array],
    bias: Optional[Array],
    output_sharding: Optional[jax.sharding.Sharding],
    dropout: float,
    rng: jax.random.PRNGKey,
    blocksize_q: int,
    blocksize_k: int,
    dtype: Optional[jnp.dtype],
    precision: Optional[lax.PrecisionLike],
) -> Tuple[Array, Tuple[Array, ...]]:
    b, h, q_len, d = query.shape
    k_len = key.shape[2]
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key and value must have the same dtype")

    out_dtype = dtype or query.dtype
    blocksize_q, blocksize_k = _compute_block_sizes(q_len, k_len, blocksize_q, blocksize_k)
    Tr = q_len // blocksize_q
    Tc = k_len // blocksize_k

    output = jnp.zeros((b, h, q_len, d), dtype=out_dtype)
    lse = jnp.full((b, h, q_len), fill_value=-jnp.inf, dtype=jnp.float32)

    output = _apply_sharding(output, output_sharding, rank=4)
    lse = _apply_sharding(lse, output_sharding, rank=3)

    mask_val = None
    if mask is not None:
        mask_val = mask.astype(bool)
    bias_val = bias

    def outer_body(i, carry):
        out_acc, lse_acc = carry
        q_i = lax.dynamic_slice_in_dim(query, i * blocksize_q, blocksize_q, axis=2)
        out_i = lax.dynamic_slice_in_dim(out_acc, i * blocksize_q, blocksize_q, axis=2)
        lse_i = lax.dynamic_slice_in_dim(lse_acc, i * blocksize_q, blocksize_q, axis=2)
        max_i = jnp.full_like(lse_i, fill_value=-jnp.inf)

        def inner_body(j, inner_carry):
            out_blk, lse_blk, max_blk = inner_carry
            k_j = lax.dynamic_slice_in_dim(key, j * blocksize_k, blocksize_k, axis=2)
            v_j = lax.dynamic_slice_in_dim(value, j * blocksize_k, blocksize_k, axis=2)

            logits = jnp.einsum(
                "bhqd,bhkd->bhqk",
                q_i,
                k_j,
                precision=precision,
            )
            logits = logits.astype(jnp.float32)

            valid = jnp.ones(logits.shape, dtype=bool)
            if bias_val is not None:
                b_i = lax.dynamic_slice_in_dim(bias_val, i * blocksize_q, blocksize_q, axis=2)
                b_ij = lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, axis=3)
                logits = logits + b_ij.astype(logits.dtype)
            if mask_val is not None:
                m_i = lax.dynamic_slice_in_dim(mask_val, i * blocksize_q, blocksize_q, axis=2)
                m_ij = lax.dynamic_slice_in_dim(m_i, j * blocksize_k, blocksize_k, axis=3)
                valid = valid & m_ij
            logits = jnp.where(valid, logits, _neg_inf(logits.dtype))

            has_valid = jnp.any(valid, axis=-1)
            block_max = jnp.where(has_valid, logits.max(axis=-1), _neg_inf(logits.dtype))
            max_new = jnp.maximum(max_blk, block_max)

            if dropout > 0.0:
                keep_prob = 1.0 - dropout
                rng_ij = jax.random.fold_in(rng, i * Tc + j)
                mask_dropout = jax.random.bernoulli(rng_ij, p=keep_prob, shape=logits.shape)
                logits = jnp.where(mask_dropout, logits / keep_prob, jnp.zeros_like(logits))

            probs = jnp.exp(logits - max_new[..., None])
            probs = jnp.where(valid, probs, jnp.zeros_like(probs))
            l_ij = probs.sum(axis=-1)

            scale = jnp.where(
                has_valid,
                jnp.where(jnp.isfinite(max_blk), jnp.exp(max_blk - max_new), jnp.zeros_like(max_new)),
                jnp.ones_like(max_new),
            )
            scaled = (out_blk * scale[..., None]).astype(out_blk.dtype)
            out_blk = jnp.where(has_valid[..., None], scaled, out_blk)
            out_blk = out_blk + jnp.einsum(
                "bhqk,bhkd->bhqd",
                probs,
                v_j,
                precision=precision,
            ).astype(out_blk.dtype)

            lse_update = jnp.where(l_ij > 0, jnp.log(l_ij), -jnp.inf) + max_new
            lse_blk = jnp.where(has_valid, jnp.logaddexp(lse_blk, lse_update), lse_blk)

            return out_blk, lse_blk, max_new

        out_i, lse_i, max_i = jax.lax.fori_loop(0, Tc, inner_body, (out_i, lse_i, max_i))
        scale = jnp.where(
            jnp.isfinite(lse_i),
            jnp.exp(max_i - lse_i),
            jnp.ones_like(lse_i),
        )
        out_i = out_i * scale[..., None]

        out_acc = lax.dynamic_update_slice_in_dim(out_acc, out_i.astype(out_acc.dtype), i * blocksize_q, axis=2)
        lse_acc = lax.dynamic_update_slice_in_dim(lse_acc, lse_i, i * blocksize_q, axis=2)
        return out_acc, lse_acc

    output, lse = jax.lax.fori_loop(0, Tr, outer_body, (output, lse))
    return output, (
        output,
        lse,
        query,
        key,
        value,
        mask_val,
        bias_val,
        rng,
        precision,
    )


def _bwd_flash_attention(
    output_sharding: Optional[jax.sharding.Sharding],
    dropout: float,
    blocksize_q: int,
    blocksize_k: int,
    dtype: Optional[jnp.dtype],
    precision: Optional[lax.PrecisionLike],
    residuals,
    grad_out: Array,
) -> Tuple[Array, ...]:
    del dtype, output_sharding
    (
        output,
        lse,
        query,
        key,
        value,
        mask,
        bias,
        rng_res,
        precision_res,
    ) = residuals
    precision = precision_res

    b, h, q_len, _ = query.shape
    k_len = key.shape[2]
    Tr = q_len // blocksize_q
    Tc = k_len // blocksize_k

    grad_out = grad_out.astype(output.dtype)
    dot = (grad_out * output).sum(axis=-1)

    grad_q = jnp.zeros_like(query)
    grad_k = jnp.zeros_like(key)
    grad_v = jnp.zeros_like(value)

    def outer_body(j, carry):
        grad_q_acc, grad_k_acc, grad_v_acc = carry
        k_j = lax.dynamic_slice_in_dim(key, j * blocksize_k, blocksize_k, axis=2)
        v_j = lax.dynamic_slice_in_dim(value, j * blocksize_k, blocksize_k, axis=2)
        grad_k_j = lax.dynamic_slice_in_dim(grad_k_acc, j * blocksize_k, blocksize_k, axis=2)
        grad_v_j = lax.dynamic_slice_in_dim(grad_v_acc, j * blocksize_k, blocksize_k, axis=2)

        def inner_body(i, inner_carry):
            grad_q_acc_in, grad_k_j_in, grad_v_j_in = inner_carry
            q_i = lax.dynamic_slice_in_dim(query, i * blocksize_q, blocksize_q, axis=2)
            grad_q_i = lax.dynamic_slice_in_dim(grad_q_acc_in, i * blocksize_q, blocksize_q, axis=2)
            grad_out_i = lax.dynamic_slice_in_dim(grad_out, i * blocksize_q, blocksize_q, axis=2)
            lse_i = lax.dynamic_slice_in_dim(lse, i * blocksize_q, blocksize_q, axis=2)
            dot_i = lax.dynamic_slice_in_dim(dot, i * blocksize_q, blocksize_q, axis=2)

            logits = jnp.einsum("bhqd,bhkd->bhqk", q_i, k_j, precision=precision).astype(jnp.float32)
            valid = jnp.ones(logits.shape, dtype=bool)
            if bias is not None:
                b_i = lax.dynamic_slice_in_dim(bias, i * blocksize_q, blocksize_q, axis=2)
                b_ij = lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, axis=3)
                logits = logits + b_ij.astype(logits.dtype)
            if mask is not None:
                m_i = lax.dynamic_slice_in_dim(mask, i * blocksize_q, blocksize_q, axis=2)
                m_ij = lax.dynamic_slice_in_dim(m_i, j * blocksize_k, blocksize_k, axis=3)
                valid = valid & m_ij

            logits = jnp.where(valid, logits, _neg_inf(logits.dtype))
            has_valid = jnp.any(valid, axis=-1)
            lse_safe = jnp.where(has_valid, lse_i, jnp.zeros_like(lse_i))

            probs = jnp.exp(logits - lse_safe[..., None])
            probs = jnp.where(valid, probs, jnp.zeros_like(probs))

            if dropout > 0.0:
                keep_prob = 1.0 - dropout
                rng_ij = jax.random.fold_in(rng_res, i * Tc + j)
                mask_dropout = jax.random.bernoulli(rng_ij, p=keep_prob, shape=probs.shape)
                probs = jnp.where(mask_dropout, probs / keep_prob, jnp.zeros_like(probs))

            grad_v_j_in = grad_v_j_in + jnp.einsum(
                "bhqk,bhqd->bhkd",
                probs,
                grad_out_i,
                precision=precision,
            ).astype(grad_v_j_in.dtype)

            grad_prob = jnp.einsum(
                "bhqd,bhkd->bhqk",
                grad_out_i,
                v_j,
                precision=precision,
            )
            grad_prob = jnp.where(valid, grad_prob, jnp.zeros_like(grad_prob))
            grad_logits = probs * (grad_prob - dot_i[..., None])

            grad_q_i = grad_q_i + jnp.einsum(
                "bhqk,bhkd->bhqd",
                grad_logits,
                k_j,
                precision=precision,
            ).astype(grad_q_i.dtype)
            grad_k_j_in = grad_k_j_in + jnp.einsum(
                "bhqk,bhqd->bhkd",
                grad_logits,
                q_i,
                precision=precision,
            ).astype(grad_k_j_in.dtype)

            grad_q_acc_in = lax.dynamic_update_slice_in_dim(
                grad_q_acc_in, grad_q_i, i * blocksize_q, axis=2
            )
            return grad_q_acc_in, grad_k_j_in, grad_v_j_in

        grad_q_acc, grad_k_j, grad_v_j = jax.lax.fori_loop(
            0, Tr, inner_body, (grad_q_acc, grad_k_j, grad_v_j)
        )

        grad_k_acc = lax.dynamic_update_slice_in_dim(grad_k_acc, grad_k_j, j * blocksize_k, axis=2)
        grad_v_acc = lax.dynamic_update_slice_in_dim(grad_v_acc, grad_v_j, j * blocksize_k, axis=2)
        return grad_q_acc, grad_k_acc, grad_v_acc

    grad_q, grad_k, grad_v = jax.lax.fori_loop(0, Tc, outer_body, (grad_q, grad_k, grad_v))

    return grad_q, grad_k, grad_v, None, None, None


_flash_attention_core.defvjp(_fwd_flash_attention, _bwd_flash_attention)


@functools.partial(
    jax.jit,
    static_argnames=(
        "dropout",
        "blocksize_q",
        "blocksize_k",
        "dtype",
        "precision",
        "output_sharding",
    ),
)
def flash_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array],
    bias: Optional[Array],
    *,
    dropout: float = 0.0,
    rng: jax.random.PRNGKey,
    blocksize_q: Optional[int] = None,
    blocksize_k: Optional[int] = None,
    dtype: Optional[jnp.dtype] = None,
    precision: Optional[lax.PrecisionLike] = None,
    output_sharding: Optional[jax.sharding.Sharding] = None,
) -> Array:
    """Low-level FlashAttention implementation on canonical 4D tensors."""
    blocksize_q, blocksize_k = _compute_block_sizes(
        query.shape[2], key.shape[2], blocksize_q, blocksize_k
    )
    return _flash_attention_core(
        query,
        key,
        value,
        mask,
        bias,
        output_sharding=output_sharding,
        dropout=dropout,
        rng=rng,
        blocksize_q=blocksize_q,
        blocksize_k=blocksize_k,
        dtype=dtype,
        precision=precision,
    )
