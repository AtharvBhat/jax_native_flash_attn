from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_native_flash_attn import flash_attention


def _random_inputs(
    rng_key: jax.random.PRNGKey,
    *,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    dim: int,
):
    key_q, key_k, key_v, key_bias, key_mask = jax.random.split(rng_key, 5)
    query = jax.random.normal(key_q, (batch, q_len, heads, dim), dtype=jnp.float32)
    key = jax.random.normal(key_k, (batch, kv_len, heads, dim), dtype=jnp.float32)
    value = jax.random.normal(key_v, (batch, kv_len, heads, dim), dtype=jnp.float32)
    bias = jax.random.normal(key_bias, (batch, heads, q_len, kv_len), dtype=jnp.float32)
    mask = jax.random.bernoulli(key_mask, p=0.8, shape=(batch, heads, q_len, kv_len))
    return query, key, value, bias, mask


def _reference_attention(*args, **kwargs):
    return jax.nn.dot_product_attention(*args, **kwargs, implementation="xla")


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_matches_reference(is_causal: bool):
    batch, heads, q_len, kv_len, dim = 2, 3, 8, 8, 16
    rng = jax.random.key(0)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    query_lengths = jnp.array([q_len, q_len - 2], dtype=jnp.int32)
    key_lengths = jnp.array([kv_len, kv_len - 3], dtype=jnp.int32)

    scale = 0.9
    local_window = (3, 2)

    ref = _reference_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        query_seq_lengths=query_lengths,
        key_value_seq_lengths=key_lengths,
        local_window_size=local_window,
    )
    attn_rng = jax.random.fold_in(jax.random.key(1), int(is_causal))
    out = flash_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        query_seq_lengths=query_lengths,
        key_value_seq_lengths=key_lengths,
        local_window_size=local_window,
        rng=attn_rng,
        blocksize_q=4,
        blocksize_k=4,
    )

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_flash_attention_gradients_match_reference():
    batch, heads, q_len, kv_len, dim = 1, 2, 8, 8, 8
    rng = jax.random.key(42)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng = jax.random.key(43)

    def forward_flash(q, k, v):
        return flash_attention(
            q,
            k,
            v,
            bias=bias,
            mask=mask,
            rng=attn_rng,
            blocksize_q=4,
            blocksize_k=4,
        ).sum()

    def forward_ref(q, k, v):
        return _reference_attention(
            q,
            k,
            v,
            bias=bias,
            mask=mask,
        ).sum()

    grad_flash = jax.grad(forward_flash, argnums=(0, 1, 2))(query, key, value)
    grad_ref = jax.grad(forward_ref, argnums=(0, 1, 2))(query, key, value)

    for grad_f, grad_r in zip(grad_flash, grad_ref):
        np.testing.assert_allclose(grad_f, grad_r, rtol=1e-4, atol=1e-4)


def test_flash_attention_backward_vjp_matches_reference():
    batch, heads, q_len, kv_len, dim = 2, 1, 4, 4, 8
    rng = jax.random.key(7)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng = jax.random.key(8)

    flash_fun = lambda q, k, v: flash_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        rng=attn_rng,
        blocksize_q=4,
        blocksize_k=4,
    )
    ref_fun = lambda q, k, v: _reference_attention(q, k, v, bias=bias, mask=mask)

    primal_out, flash_vjp = jax.vjp(flash_fun, query, key, value)
    ref_out, ref_vjp = jax.vjp(ref_fun, query, key, value)

    np.testing.assert_allclose(primal_out, ref_out, rtol=1e-4, atol=1e-4)

    cotangent = jax.random.normal(rng, primal_out.shape, dtype=primal_out.dtype)
    flash_grads = flash_vjp(cotangent)
    ref_grads = ref_vjp(cotangent)

    for grad_f, grad_r in zip(flash_grads, ref_grads):
        np.testing.assert_allclose(grad_f, grad_r, rtol=1e-4, atol=1e-4)


def test_flash_attention_supports_scale_only():
    batch, heads, q_len, kv_len, dim = 2, 1, 4, 4, 8
    rng = jax.random.key(123)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng = jax.random.key(124)

    ref = _reference_attention(
        query,
        key,
        value,
        scale=None,
    )
    out = flash_attention(
        query,
        key,
        value,
        rng=attn_rng,
        blocksize_q=4,
        blocksize_k=4,
    )

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_flash_attention_respects_input_sharding():
    devices = jax.devices()
    if not devices:
        pytest.skip("No JAX devices available for sharding test")

    mesh_size = min(2, len(devices))
    mesh_devices = np.array(devices[:mesh_size]).reshape((mesh_size,))
    mesh_axes = ("data",)

    batch, heads, q_len, kv_len, dim = 4, 2, 8, 8, 16
    host_query = jnp.arange(batch * q_len * heads * dim, dtype=jnp.float32).reshape(
        batch, q_len, heads, dim
    )
    host_key = jnp.arange(batch * kv_len * heads * dim, dtype=jnp.float32).reshape(
        batch, kv_len, heads, dim
    )
    host_value = jnp.arange(batch * kv_len * heads * dim, dtype=jnp.float32).reshape(
        batch, kv_len, heads, dim
    )

    mesh = jax.sharding.Mesh(mesh_devices, mesh_axes)
    with mesh:
        sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec("data", None, None, None),
        )
        query = jax.device_put(host_query, sharding)
        key = jax.device_put(host_key, sharding)
        value = jax.device_put(host_value, sharding)
        attn_rng = jax.random.key(0)

        out = flash_attention(
            query,
            key,
            value,
            rng=attn_rng,
            blocksize_q=4,
            blocksize_k=4,
            output_sharding=sharding,
        )

    assert hasattr(out, "sharding")
    assert out.sharding == query.sharding


def test_flash_attention_rejects_mismatched_output_sharding():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Need at least two devices to construct mismatched sharding")

    sharding_query = jax.sharding.SingleDeviceSharding(devices[0])
    sharding_other = jax.sharding.SingleDeviceSharding(devices[1])

    data = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    query = jax.device_put(data, sharding_query)
    key = jax.device_put(data, sharding_query)
    value = jax.device_put(data, sharding_query)

    with pytest.raises(AssertionError, match="output_sharding must match"):
        flash_attention(
            query,
            key,
            value,
            rng=jax.random.key(0),
            output_sharding=sharding_other,
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_flash_attention_matches_reference_dtypes(dtype):
    batch, heads, q_len, kv_len, dim = 2, 2, 16, 16, 32
    rng = jax.random.key(3)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng = jax.random.key(4)

    query = query.astype(dtype)
    key = key.astype(dtype)
    value = value.astype(dtype)
    bias = bias.astype(dtype)

    ref = _reference_attention(query, key, value, bias=bias, mask=mask, scale=None)
    out = flash_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        blocksize_q=8,
        blocksize_k=8,
        dtype=dtype,
        rng=attn_rng,
    )

    np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_flash_attention_jittable():
    batch, heads, q_len, kv_len, dim = 1, 4, 32, 32, 16
    rng = jax.random.key(21)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng = jax.random.key(22)

    flash_jitted = jax.jit(flash_attention, static_argnames=("is_causal",))
    ref_jitted = jax.jit(_reference_attention, static_argnames=("is_causal",))

    out = flash_jitted(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        is_causal=True,
        rng=attn_rng,
    )
    ref = ref_jitted(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        is_causal=True,
    )

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_flash_attention_dropout_requires_rng():
    query = jnp.ones((1, 4, 2, 8))
    key = jnp.ones((1, 4, 2, 8))
    value = jnp.ones((1, 4, 2, 8))

    with pytest.raises(ValueError, match="An rng key must be provided"):
        flash_attention(
            query,
            key,
            value,
            dropout_rate=0.1,
            rng=None,
        )


def test_flash_attention_dropout_reproducible_with_same_rng():
    query = jax.random.normal(jax.random.key(0), (1, 8, 2, 16))
    key = jax.random.normal(jax.random.key(1), (1, 8, 2, 16))
    value = jax.random.normal(jax.random.key(2), (1, 8, 2, 16))
    rng = jax.random.key(3)

    out1 = flash_attention(
        query,
        key,
        value,
        dropout_rate=0.2,
        rng=rng,
    )
    out2 = flash_attention(
        query,
        key,
        value,
        dropout_rate=0.2,
        rng=rng,
    )

    np.testing.assert_allclose(out1, out2)


def test_flash_attention_local_window_int_matches_tuple():
    batch, heads, q_len, kv_len, dim = 1, 2, 16, 16, 8
    rng = jax.random.key(99)
    query, key, value, bias, mask = _random_inputs(
        rng, batch=batch, heads=heads, q_len=q_len, kv_len=kv_len, dim=dim
    )
    attn_rng_int = jax.random.key(100)
    attn_rng_tuple = jax.random.fold_in(attn_rng_int, 1)

    out_int = flash_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        local_window_size=3,
        blocksize_q=8,
        blocksize_k=8,
        rng=attn_rng_int,
    )
    out_tuple = flash_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        local_window_size=(3, 3),
        blocksize_q=8,
        blocksize_k=8,
        rng=attn_rng_tuple,
    )

    np.testing.assert_allclose(out_int, out_tuple)
