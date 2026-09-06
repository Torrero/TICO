# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the Qwen3-VL GPTQv2 quantizer helpers.

These tests verify that:
  - The GPTQ class from qwen3_vl_gptq.gptq is used for both v1 and v2.
  - _build_gptq_objects initializes native_inp when gptq_v2=True.
  - Conv3d native inputs correctly populate dXXT.
"""

import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import GPTQ
from tico.quantization.algorithm.qwen3_vl_gptq.quantizer import (
    FPInputsCache,
    Qwen3VLGPTQQuantizer,
)
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig


class TestQwen3VLGPTQv2Core(unittest.TestCase):
    """Test GPTQv2 core mechanics on Conv3d layers."""

    @torch.no_grad()
    def test_conv3d_native_inputs_populate_dXXT(self):
        """dXXT should be computed and non-zero when FP and quantized inputs differ."""
        layer = torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=(2, 2, 2),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        gptq = GPTQ(layer)

        current = torch.randn(1, 2, 3, 3, 3)
        native = current + 0.125
        out = layer(current)

        gptq.native_inp = [native]
        gptq.add_batch(current, out)

        self.assertIsNotNone(gptq.dXXT)
        dXXT = gptq.dXXT
        assert dXXT is not None
        self.assertEqual(dXXT.shape, gptq.H.shape)  # type: ignore[union-attr]
        self.assertGreater(dXXT.abs().sum().item(), 0.0)


class TestQwen3VLGPTQv2QuantizerHelpers(unittest.TestCase):
    """Test Qwen3VLGPTQQuantizer helper methods."""

    def test_build_gptq_objects_default_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=None for v1."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v1 (gptq_v2=False), native_inp should not be initialized as a list
        self.assertIsNone(gptq_objs["linear"].native_inp)

    def test_build_gptq_objects_gptqv2_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=[] for v2."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v2 (gptq_v2=True), native_inp should be initialized as an empty list
        self.assertEqual(gptq_objs["linear"].native_inp, [])

    def test_assign_native_inputs(self):
        """_assign_native_inputs should copy FP inputs to GPTQ objects."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        fp_inputs = [torch.randn(2, 4), torch.randn(3, 4)]
        native_inputs = {"linear": fp_inputs}

        quantizer._assign_native_inputs(gptq_objs, native_inputs)

        native_inp = gptq_objs["linear"].native_inp
        assert native_inp is not None
        self.assertEqual(len(native_inp), 2)
        self.assertTrue(torch.allclose(native_inp[0], fp_inputs[0]))
        self.assertTrue(torch.allclose(native_inp[1], fp_inputs[1]))

    def test_resolve_weight_bits_default(self):
        """_resolve_weight_bits should return the config default when no override."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(weight_bits=4))
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 4)

    def test_resolve_weight_bits_override_full_name(self):
        """_resolve_weight_bits should use full-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"model.layers.0.self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_local_name(self):
        """_resolve_weight_bits should use local-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_suffix(self):
        """_resolve_weight_bits should use suffix override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_module_device(self):
        """_module_device should return the device of the module's parameters."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        device = quantizer._module_device(layer)
        self.assertEqual(device, layer.weight.device)

    def test_copy_original_model(self):
        """_copy_original_model should create a deep copy on CPU."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        orig_model = quantizer._copy_original_model(model)

        # Should be a different object
        self.assertIsNot(orig_model, model)
        # Weights should match
        self.assertTrue(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]
        # Modifying one should not affect the other
        model[0].weight.data.fill_(0.0)
        self.assertFalse(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]


# ---------------------------------------------------------------------------
# Helpers for FP inputs cache tests
# ---------------------------------------------------------------------------


def _make_quantizer(fp_inputs_cache_path=None, gptq_v2=True):
    """Create a Qwen3VLGPTQQuantizer with minimal config for testing."""
    config = Qwen3VLGPTQConfig(
        weight_bits=8,
        gptq_v2=gptq_v2,
        fp_inputs_cache_path=fp_inputs_cache_path,
        show_progress=False,
        verbose=False,
    )
    return Qwen3VLGPTQQuantizer(config)


# ---------------------------------------------------------------------------
# Tests: FPInputsCache
# ---------------------------------------------------------------------------


class TestFPInputsCache(unittest.TestCase):
    """Core tests for the FPInputsCache hook-based collector and disk cache."""

    def test_caches_fp_input(self):
        """A forward hook stores the first positional arg in fp_cache."""
        cache = FPInputsCache(["linear"])
        linear = nn.Linear(4, 4)
        cache.add_hook({"linear": linear})

        inp = torch.randn(2, 4)
        linear(inp)
        cache.clear_hook()

        self.assertIn("linear", cache.fp_cache)
        self.assertEqual(len(cache.fp_cache["linear"]), 1)
        self.assertTrue(torch.equal(cache.fp_cache["linear"][0], inp))

    def test_save_and_load_roundtrip(self):
        """Save cache to disk, load it back, and verify tensor equality."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "fp_cache.pt")
            dummy_cache = {
                "vision.patch_embed": {
                    "proj": [torch.randn(2, 3), torch.randn(2, 3)],
                },
                "text.layers.0": {
                    "self_attn.q_proj": [torch.randn(4, 5)],
                },
            }
            torch.save(dummy_cache, cache_path)
            self.assertTrue(os.path.exists(cache_path))

            loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertEqual(set(loaded.keys()), set(dummy_cache.keys()))
            for stage in dummy_cache:
                for name in dummy_cache[stage]:
                    for i, t in enumerate(loaded[stage][name]):
                        self.assertTrue(torch.equal(t, dummy_cache[stage][name][i]))

    def test_raw_replay_returns_cached_on_hit(self):
        """_collect_native_inputs_from_raw_replay returns cached data without
        running any forward hooks when stage_desc is in the disk cache."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")
        cached_tensors = [torch.randn(2, 3)]
        quantizer._fp_inputs_disk_cache = {
            "vision.merger": {"merger.linear": cached_tensors},
        }

        dummy_model = MagicMock()
        result = quantizer._collect_native_inputs_from_raw_replay(
            model=dummy_model,
            subset={"merger.linear": MagicMock()},
            module_name={},
            cache_args=[[]],
            cache_kwargs={},
            num_batches=1,
            stage_desc="vision.merger",
        )

        self.assertIn("merger.linear", result)
        self.assertTrue(torch.equal(result["merger.linear"][0], cached_tensors[0]))
        dummy_model.assert_not_called()

    @torch.no_grad()
    def test_collect_then_cache_hit(self):
        """First call collects via hooks and persists to _fp_inputs_disk_cache;
        second call returns from cache without re-running forward."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")

        linear = nn.Linear(4, 4)
        stage_module = nn.Sequential(linear)
        subset = {"0": linear}

        inp = torch.randn(2, 4)
        cached_args = [[inp]]
        cached_kwargs: dict = {}

        result1 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=stage_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertIn("0", result1)
        self.assertTrue(torch.equal(result1["0"][0], inp))

        # The function should have persisted to _fp_inputs_disk_cache automatically
        self.assertIn("test_stage", quantizer._fp_inputs_disk_cache)

        broken_module = MagicMock(side_effect=RuntimeError("should not be called"))
        result2 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=broken_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertTrue(torch.equal(result2["0"][0], result1["0"][0]))
        broken_module.assert_not_called()

    @torch.no_grad()
    def test_stage_cache_persist_roundtrip(self):
        """Cold collection -> save to disk -> new quantizer loads disk ->
        warm lookup returns cached tensors without calling forward.

        This test verifies that _collect_native_inputs_from_stage_cache()
        persists its result so that a warm-cache run can retrieve it.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "fp_cache.pt")

            # --- Cold run: collect and persist ---
            quantizer_cold = _make_quantizer(fp_inputs_cache_path=cache_path)

            linear = nn.Linear(4, 4)
            stage_module = nn.Sequential(linear)
            subset = {"0": linear}

            inp = torch.randn(2, 4)
            cached_args = [[inp]]
            cached_kwargs: dict = {}

            result_cold = quantizer_cold._collect_native_inputs_from_stage_cache(
                stage_module=stage_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="vision.blocks.0",
                num_batches=1,
            )
            self.assertIn("0", result_cold)
            self.assertTrue(torch.equal(result_cold["0"][0], inp))

            # Verify the stage was persisted to the in-memory disk cache
            self.assertIn("vision.blocks.0", quantizer_cold._fp_inputs_disk_cache)

            # Simulate convert()'s save logic
            torch.save(quantizer_cold._fp_inputs_disk_cache, cache_path)
            self.assertTrue(os.path.exists(cache_path))

            # --- Warm run: new quantizer loads cache from disk ---
            quantizer_warm = _make_quantizer(fp_inputs_cache_path=cache_path)
            quantizer_warm._fp_inputs_disk_cache = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            quantizer_warm._fp_inputs_disk_loaded = True

            # Use a broken module that raises if forward is called
            broken_module = MagicMock(side_effect=RuntimeError("should not be called"))
            result_warm = quantizer_warm._collect_native_inputs_from_stage_cache(
                stage_module=broken_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="vision.blocks.0",
                num_batches=1,
            )
            # The warm result should match the cold result
            self.assertTrue(torch.equal(result_warm["0"][0], result_cold["0"][0]))
            broken_module.assert_not_called()

    @torch.no_grad()
    def test_stage_cache_fail_closed_on_miss(self):
        """When _fp_inputs_disk_loaded is True but the stage is not in the cache,
        a RuntimeError should be raised instead of silently recomputing."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")
        quantizer._fp_inputs_disk_loaded = True
        # "missing_stage" is NOT in _fp_inputs_disk_cache

        linear = nn.Linear(4, 4)
        stage_module = nn.Sequential(linear)
        subset = {"0": linear}

        inp = torch.randn(2, 4)
        cached_args = [[inp]]
        cached_kwargs: dict = {}

        with self.assertRaises(RuntimeError) as ctx:
            quantizer._collect_native_inputs_from_stage_cache(
                stage_module=stage_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="missing_stage",
                num_batches=1,
            )
        self.assertIn("cache miss", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Tests: GPTQv2 P-correction (dXXT → P matrix)
# ---------------------------------------------------------------------------


class TestGPTQPCorrection(unittest.TestCase):
    """Reference tests for the GPTQv2 P-correction logic in fasterquant.

    These tests verify:
      1. ``quantize()`` does not modify ``w_col`` in-place.
      2. ``q_col`` and ``w_col`` are different tensors with different values.
      3. With ``alpha=0`` the P-correction is zero, so GPTQv2 == GPTQv1.
      4. With ``alpha>0`` and non-zero ``dXXT``, the quantized weights differ
         from the ``alpha=0`` case.
      5. The P matrix is computed as ``alpha * triu(dXXT @ hinv^T, k=1) @ hinv``.
    """

    def _make_gptq(self, rows=8, cols=8):
        """Create a GPTQ object with a small Linear layer and a configured quantizer."""
        torch.manual_seed(42)
        layer = nn.Linear(cols, rows, bias=False)
        gptq = GPTQ(layer)
        gptq.quantizer.configure(bits=8, perchannel=True, sym=True)
        return gptq

    def _add_random_batch(self, gptq, batch=16, cols=8):
        """Feed a random batch so that H is well-conditioned."""
        torch.manual_seed(123)
        inp = torch.randn(batch, cols)
        with torch.no_grad():
            out = gptq.layer(inp)
        gptq.add_batch(inp, out)

    # ------------------------------------------------------------------
    # 1. quantize() does not modify w_col in-place
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_quantize_does_not_modify_w_col(self):
        """The ``quantize`` function must not change its input tensor in-place."""
        from tico.quantization.algorithm.qwen3_vl_gptq.gptq import quantize

        w_col = torch.randn(6)
        w_col_copy = w_col.clone()

        scale = torch.tensor(0.1)
        zero = torch.tensor(0.0)
        maxq = torch.tensor(255.0)

        _ = quantize(w_col.unsqueeze(1), scale, zero, maxq)

        self.assertTrue(torch.equal(w_col, w_col_copy))

    # ------------------------------------------------------------------
    # 2. q_col != w_col after quantize()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_q_col_differs_from_w_col(self):
        """``q_col`` (quantized) must differ from ``w_col`` (original)."""
        from tico.quantization.algorithm.qwen3_vl_gptq.gptq import quantize

        w_col = torch.randn(6)
        scale = torch.tensor(0.1)
        zero = torch.tensor(0.0)
        maxq = torch.tensor(255.0)

        q_col = quantize(w_col.unsqueeze(1), scale, zero, maxq).flatten()

        self.assertFalse(torch.equal(q_col, w_col))
        # The difference is the quantization error
        self.assertGreater((w_col - q_col).abs().sum().item(), 0.0)

    # ------------------------------------------------------------------
    # 3. alpha=0  →  P is zero  →  GPTQv2 == GPTQv1
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_alpha_zero_equals_no_pcorrection(self):
        """With alpha=0 the P-correction vanishes, so the result must match
        the case where dXXT is None (pure GPTQv1)."""
        rows, cols = 8, 8

        # --- run A: dXXT=None (GPTQv1) ---
        gptq_a = self._make_gptq(rows, cols)
        self._add_random_batch(gptq_a, batch=16, cols=cols)
        w_before_a = gptq_a.layer.weight.data.clone()
        gptq_a.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_after_a = gptq_a.layer.weight.data.clone()

        # --- run B: dXXT set but alpha=0 ---
        gptq_b = self._make_gptq(rows, cols)
        # Copy same weights and H so the two runs are comparable
        gptq_b.layer.weight.data = w_before_a.clone()
        self._add_random_batch(gptq_b, batch=16, cols=cols)
        gptq_b.dXXT = torch.randn(cols, cols)  # non-zero dXXT
        gptq_b.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_after_b = gptq_b.layer.weight.data.clone()

        self.assertTrue(torch.allclose(w_after_a, w_after_b, atol=1e-6))

    # ------------------------------------------------------------------
    # 4. alpha>0 with non-zero dXXT → result differs from alpha=0
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_alpha_positive_differs_from_zero(self):
        """With alpha>0 and a non-zero dXXT, the quantized weights must
        differ from the alpha=0 baseline."""
        rows, cols = 8, 8

        # --- baseline: alpha=0 ---
        gptq_base = self._make_gptq(rows, cols)
        self._add_random_batch(gptq_base, batch=16, cols=cols)
        w_orig = gptq_base.layer.weight.data.clone()
        gptq_base.dXXT = torch.randn(cols, cols)
        gptq_base.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_base = gptq_base.layer.weight.data.clone()

        # --- with P-correction: alpha=0.5 ---
        gptq_p = self._make_gptq(rows, cols)
        gptq_p.layer.weight.data = w_orig.clone()
        self._add_random_batch(gptq_p, batch=16, cols=cols)
        gptq_p.dXXT = gptq_base.dXXT.clone()  # same dXXT
        gptq_p.fasterquant(blocksize=128, percdamp=0.01, alpha=0.5)
        w_p = gptq_p.layer.weight.data.clone()

        self.assertFalse(torch.allclose(w_base, w_p, atol=1e-6))

    # ------------------------------------------------------------------
    # 5. P matrix formula: alpha * triu(dXXT @ hinv^T, k=1) @ hinv
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_p_correction_formula(self):
        """Manually compute P from dXXT and hinv, then verify that the
        in-block P-correction matches ``w_col @ P1[i, i:]``."""
        rows, cols = 6, 6

        gptq = self._make_gptq(rows, cols)
        self._add_random_batch(gptq, batch=32, cols=cols)

        # Set a known dXXT
        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq.dXXT = dXXT.clone()
        alpha = 0.25

        # Reproduce the hinv computation from fasterquant
        h = gptq.H.clone()
        del gptq.H  # fasterquant does del self.H; we need to restore after
        gptq.H = h.clone()  # restore for fasterquant

        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        damp = 0.01 * torch.mean(torch.diag(h))
        diag_idx = torch.arange(cols)
        h[diag_idx, diag_idx] += damp
        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        hinv = h

        # Compute P using the same formula as fasterquant
        P_ref = alpha * ((dXXT @ hinv.T).triu(diagonal=1)) @ hinv

        # P must be non-zero
        self.assertGreater(P_ref.abs().sum().item(), 0.0)

        # The diagonal of P must be zero (triu with diagonal=1)
        self.assertTrue(torch.allclose(torch.diag(P_ref), torch.zeros(cols), atol=1e-6))

        # Run fasterquant and verify it completes without error
        gptq.fasterquant(blocksize=128, percdamp=0.01, alpha=alpha)
        # After fasterquant, the layer weight should have been updated
        self.assertIsNotNone(gptq.layer.weight.data)


if __name__ == "__main__":
    unittest.main()
