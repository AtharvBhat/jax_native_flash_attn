"""Micro-benchmarks comparing FlashAttention variants."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp

from jax_native_flash_attn import flash_attention


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch: int
    q_len: int
    kv_len: int
    heads: int
    head_dim: int

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.batch, self.q_len, self.heads, self.head_dim)


DEFAULT_CASES: Tuple[BenchmarkCase, ...] = (
    BenchmarkCase("small", batch=4, q_len=128, kv_len=128, heads=4, head_dim=64),
    BenchmarkCase("medium", batch=8, q_len=256, kv_len=256, heads=8, head_dim=64),
    BenchmarkCase("long-context", batch=4, q_len=512, kv_len=512, heads=16, head_dim=64),
)


def _prepare_inputs(case: BenchmarkCase) -> Tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(0)
    key_q, key_k, key_v = jax.random.split(key, 3)
    query = jax.random.normal(key_q, case.shape, dtype=jnp.float32)
    key = jax.random.normal(key_k, (case.batch, case.kv_len, case.heads, case.head_dim), dtype=jnp.float32)
    value = jax.random.normal(key_v, (case.batch, case.kv_len, case.heads, case.head_dim), dtype=jnp.float32)
    return query, key, value


def _measure(callable_fn, warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        callable_fn().block_until_ready()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        result = callable_fn()
        result.block_until_ready()
        samples.append(time.perf_counter() - start)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def run_case(case: BenchmarkCase, warmup: int, iters: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    query, key, value = _prepare_inputs(case)

    rng = jax.random.key(42)

    def flash_fn(q, k, v):
        return flash_attention(q, k, v, rng=rng, blocksize_q=128, blocksize_k=128)

    flash_jitted = jax.jit(flash_fn)

    def ref_fn(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, implementation="xla")

    ref_jitted = jax.jit(ref_fn)

    flash_time = _measure(lambda: flash_jitted(query, key, value), warmup, iters)
    ref_time = _measure(lambda: ref_jitted(query, key, value), warmup, iters)
    return flash_time, ref_time


def benchmark(cases: Iterable[BenchmarkCase], warmup: int, iters: int) -> None:
    print(f"Running benchmarks with warmup={warmup}, iters={iters}")
    for case in cases:
        flash_time, ref_time = run_case(case, warmup, iters)
        flash_mean, flash_std = flash_time
        ref_mean, ref_std = ref_time
        speedup = ref_mean / flash_mean if flash_mean else float("nan")
        print(
            f"{case.name:>12}: flash_attention {flash_mean*1e3:8.2f}±{flash_std*1e3:4.2f} ms | "
            f"dot_product {ref_mean*1e3:8.2f}±{ref_std*1e3:4.2f} ms | "
            f"speedup ×{speedup:4.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention vs. dot_product_attention")
    parser.add_argument("--cases", nargs="*", choices=[case.name for case in DEFAULT_CASES], help="Subset of benchmark cases to run")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations to run before timing")
    parser.add_argument("--iters", type=int, default=10, help="Number of timed iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases:
        selected = tuple(case for case in DEFAULT_CASES if case.name in args.cases)
    else:
        selected = DEFAULT_CASES
    benchmark(selected, args.warmup, args.iters)


if __name__ == "__main__":
    main()
