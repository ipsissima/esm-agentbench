"""Tests for Multi-Scale Spectral Monitoring (Phase 4 AgentX).

This module tests the Adaptive Multi-Scale Spectral Verification system that:
- Local Coherence (Micro-Monitor): Sliding window SVD for sub-task consistency
- Global Alignment (Macro-Monitor): Semantic check against task embedding
- Dual-metric verdict logic: Handles context switches without false positives
"""
import numpy as np
import pytest

from certificates.make_certificate import (
    compute_adaptive_certificate,
    _compute_local_coherence,
    _compute_global_alignment,
    segment_trace_by_jump,
)


class TestLocalCoherence:
    """Tests for Local Coherence (Micro-Monitor) computation."""

    def test_stable_trace_has_low_local_bound(self):
        """A low-variance trace should have low local spectral bound."""
        rng = np.random.default_rng(42)
        # Create a stable trace with minimal variation
        base = rng.normal(size=(1, 64))
        stable_trace = np.tile(base, (30, 1)) + rng.normal(size=(30, 64)) * 0.01

        local = _compute_local_coherence(stable_trace, window_size=20, r=5)

        assert local["local_spectral_bound"] < 0.3
        assert local["local_pca_explained"] > 0.9
        assert local["local_tail_energy"] < 0.1

    def test_chaotic_trace_has_high_local_bound(self):
        """A high-variance random trace should have higher local bound."""
        rng = np.random.default_rng(42)
        chaotic_trace = rng.normal(size=(30, 64))

        local = _compute_local_coherence(chaotic_trace, window_size=20, r=5)

        # Should have higher bound than stable trace
        assert local["local_spectral_bound"] > 0.1
        assert local["local_pca_explained"] < 0.9

    def test_window_respects_size(self):
        """Local coherence should only analyze the last window_size steps."""
        rng = np.random.default_rng(42)
        trace = rng.normal(size=(50, 32))

        local = _compute_local_coherence(trace, window_size=15, r=5)

        # Window should be last 15 steps
        assert local["window_start"] == 35
        assert local["window_end"] == 50

    def test_short_trace_handling(self):
        """Should handle traces shorter than window size."""
        rng = np.random.default_rng(42)
        short_trace = rng.normal(size=(5, 16))

        local = _compute_local_coherence(short_trace, window_size=20, r=3)

        # Should use entire trace as window
        assert local["window_start"] == 0
        assert local["window_end"] == 5
        assert 0 <= local["local_spectral_bound"] <= 2.0

    def test_edge_case_single_step(self):
        """Single-step trace should return conservative values."""
        single_step = np.array([[1.0, 2.0, 3.0]])

        local = _compute_local_coherence(single_step, window_size=20, r=3)

        assert local["local_spectral_bound"] == 1.0  # Conservative
        assert local["local_pca_explained"] == 0.0


class TestGlobalAlignment:
    """Tests for Global Alignment (Macro-Monitor) computation."""

    def test_aligned_trace_has_low_drift(self):
        """A trace that stays near task embedding should have low drift."""
        rng = np.random.default_rng(42)
        task_emb = rng.normal(size=64)
        # Create trace that stays close to task embedding
        aligned_trace = np.tile(task_emb, (20, 1)) + rng.normal(size=(20, 64)) * 0.1

        global_align = _compute_global_alignment(aligned_trace, task_emb, window_size=10)

        assert global_align["global_semantic_drift"] < 0.3
        assert global_align["window_semantic_drift"] < 0.3
        assert global_align["cumulative_semantic_drift"] < 0.3

    def test_drifting_trace_has_high_drift(self):
        """A trace that diverges from task embedding should have high drift."""
        rng = np.random.default_rng(42)
        task_emb = np.ones(64)  # Fixed task embedding
        # Create trace that moves away from task
        drifting_trace = rng.normal(size=(20, 64))  # Random, uncorrelated

        global_align = _compute_global_alignment(drifting_trace, task_emb, window_size=10)

        # Should have higher drift
        assert global_align["global_semantic_drift"] > 0.5

    def test_slow_drift_detection(self):
        """Should catch slow drift where later steps diverge more."""
        rng = np.random.default_rng(42)
        task_emb = np.ones(64)

        # First half aligned, second half divergent
        aligned_part = np.tile(task_emb, (10, 1)) + rng.normal(size=(10, 64)) * 0.05
        divergent_part = rng.normal(size=(10, 64)) * 2.0
        slow_drift_trace = np.vstack([aligned_part, divergent_part])

        global_align = _compute_global_alignment(slow_drift_trace, task_emb, window_size=10)

        # Window drift (recent) should be higher than cumulative
        assert global_align["window_semantic_drift"] > global_align["cumulative_semantic_drift"]

    def test_uses_first_step_when_no_task_embedding(self):
        """Should use first step as reference when no task embedding provided."""
        rng = np.random.default_rng(42)
        base = rng.normal(size=32)
        trace = np.tile(base, (15, 1)) + rng.normal(size=(15, 32)) * 0.05

        global_align = _compute_global_alignment(trace, task_embedding=None, window_size=10)

        # Should have low drift (all steps similar to first)
        assert global_align["global_semantic_drift"] < 0.3


class TestAdaptiveCertificate:
    """Tests for the main compute_adaptive_certificate function."""

    def test_stable_aligned_trace_passes(self):
        """A stable, task-aligned trace should pass."""
        rng = np.random.default_rng(42)
        task_emb = rng.normal(size=64)
        # Stable trace near task embedding
        stable_trace = np.tile(task_emb, (30, 1)) + rng.normal(size=(30, 64)) * 0.02

        cert = compute_adaptive_certificate(
            stable_trace,
            task_embedding=task_emb,
            window_size=20,
            local_bound_threshold=0.5,
            global_drift_threshold=0.7,
        )

        assert cert["multi_scale_verdict"] == "PASS"
        assert cert["local_spectral_bound"] < 0.5
        assert cert["global_semantic_drift"] < 0.7
        assert "Local stable" in cert["reasoning"]
        assert "Global aligned" in cert["reasoning"]

    def test_chaotic_trace_fails_instability(self):
        """A chaotic trace should fail with FAIL_INSTABILITY."""
        rng = np.random.default_rng(42)
        task_emb = rng.normal(size=64)
        # Highly chaotic trace
        chaotic_trace = rng.normal(size=(30, 64)) * 5.0

        cert = compute_adaptive_certificate(
            chaotic_trace,
            task_embedding=task_emb,
            window_size=20,
            local_bound_threshold=0.3,  # Tight threshold
            global_drift_threshold=0.7,
        )

        assert cert["multi_scale_verdict"] == "FAIL_INSTABILITY"
        assert "UNSTABLE" in cert["reasoning"]

    def test_goal_drift_fails(self):
        """A locally stable but globally divergent trace should fail with FAIL_GOAL_DRIFT."""
        rng = np.random.default_rng(42)
        task_emb = np.ones(64)

        # Create a trace that is locally coherent but diverges from task
        # The trick: make each step similar to neighbors (local stability)
        # but all far from task embedding (global drift)
        divergent_base = rng.normal(size=64) * 2.0  # Far from task
        locally_stable_trace = np.tile(divergent_base, (30, 1)) + rng.normal(size=(30, 64)) * 0.02

        cert = compute_adaptive_certificate(
            locally_stable_trace,
            task_embedding=task_emb,
            window_size=20,
            local_bound_threshold=0.5,
            global_drift_threshold=0.5,  # Tight threshold for drift
        )

        assert cert["multi_scale_verdict"] == "FAIL_GOAL_DRIFT"
        assert "GOAL DRIFT" in cert["reasoning"]
        assert "forgotten" in cert["reasoning"].lower()

    def test_context_switch_handling(self):
        """Should detect context switches (segments) in the trace."""
        rng = np.random.default_rng(42)

        # Create a trace with a clear context switch
        phase1 = rng.normal(size=(15, 32)) + np.array([1.0] * 32)  # Planning mode
        phase2 = rng.normal(size=(15, 32)) + np.array([-1.0] * 32)  # Coding mode

        # Add a jump between phases
        trace = np.vstack([phase1, phase2])

        cert = compute_adaptive_certificate(
            trace,
            task_embedding=None,
            window_size=10,
            jump_threshold=0.2,
        )

        # Should detect segments
        assert "segments" in cert
        assert len(cert["segments"]) >= 1
        assert "num_context_switches" in cert

    def test_output_structure(self):
        """Verify all expected fields are present in output."""
        rng = np.random.default_rng(42)
        trace = rng.normal(size=(25, 32))
        task_emb = rng.normal(size=32)

        cert = compute_adaptive_certificate(trace, task_embedding=task_emb)

        # Top-level fields
        assert "multi_scale_verdict" in cert
        assert "local_spectral_bound" in cert
        assert "global_semantic_drift" in cert
        assert "local_coherence" in cert
        assert "global_alignment" in cert
        assert "segments" in cert
        assert "active_segment" in cert
        assert "reasoning" in cert

        # Local coherence subfields
        lc = cert["local_coherence"]
        assert "local_spectral_bound" in lc
        assert "local_residual" in lc
        assert "local_pca_explained" in lc
        assert "window_start" in lc
        assert "window_end" in lc

        # Global alignment subfields
        ga = cert["global_alignment"]
        assert "global_semantic_drift" in ga
        assert "window_semantic_drift" in ga
        assert "cumulative_semantic_drift" in ga
        assert "max_semantic_drift" in ga

        # Base certificate compatibility
        assert "theoretical_bound" in cert
        assert "residual" in cert
        assert "pca_explained" in cert

    def test_short_trace_handling(self):
        """Should handle very short traces gracefully."""
        rng = np.random.default_rng(42)
        short_trace = rng.normal(size=(1, 16))

        cert = compute_adaptive_certificate(short_trace, window_size=20)

        assert cert["multi_scale_verdict"] == "PASS"
        assert "Insufficient" in cert["reasoning"]

    def test_window_size_parameter(self):
        """Window size should affect local coherence analysis."""
        rng = np.random.default_rng(42)
        trace = rng.normal(size=(50, 32))

        cert_small_window = compute_adaptive_certificate(trace, window_size=5)
        cert_large_window = compute_adaptive_certificate(trace, window_size=30)

        # Different windows should give different local coherence window bounds
        lc_small = cert_small_window["local_coherence"]
        lc_large = cert_large_window["local_coherence"]

        assert lc_small["window_end"] - lc_small["window_start"] <= 5
        assert lc_large["window_end"] - lc_large["window_start"] <= 30


class TestSegmentTraceByJump:
    """Tests for the segment_trace_by_jump helper."""

    def test_no_jumps_single_segment(self):
        """Trace with no jumps should be single segment."""
        residuals = [0.1, 0.12, 0.11, 0.13, 0.1]
        segments = segment_trace_by_jump(residuals, jump_threshold=0.5)

        assert len(segments) == 1
        assert segments[0] == (0, 5)

    def test_single_jump_two_segments(self):
        """Trace with one jump should have two segments."""
        residuals = [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]  # Jump at index 3
        segments = segment_trace_by_jump(residuals, jump_threshold=0.5)

        assert len(segments) == 2
        assert segments[0][1] == 3  # First segment ends at jump
        assert segments[1][0] == 3  # Second segment starts at jump

    def test_multiple_jumps(self):
        """Trace with multiple jumps should have multiple segments."""
        residuals = [0.1, 0.1, 0.8, 0.8, 0.2, 0.2]  # Jumps at indices 2 and 4
        segments = segment_trace_by_jump(residuals, jump_threshold=0.5)

        assert len(segments) == 3

    def test_empty_residuals(self):
        """Empty residuals should return empty segment."""
        segments = segment_trace_by_jump([], jump_threshold=0.5)
        assert segments == [(0, 0)]

    def test_handles_none_values(self):
        """Should handle None values in residuals."""
        residuals = [0.1, None, 0.1, 0.1]
        segments = segment_trace_by_jump(residuals, jump_threshold=0.5)

        # Should treat None as 0.0 and not crash
        assert len(segments) >= 1


class TestMultiScaleIntegration:
    """Integration tests for multi-scale monitoring in realistic scenarios."""

    def test_planning_to_coding_transition(self):
        """Simulate a realistic Planning → Coding transition."""
        rng = np.random.default_rng(42)

        # Task: "Implement a sorting function"
        task_emb = rng.normal(size=64)

        # Planning phase: discussing approach (close to task)
        planning = np.tile(task_emb, (15, 1)) + rng.normal(size=(15, 64)) * 0.1

        # Coding phase: writing code (still aligned but different subspace)
        coding_center = task_emb + rng.normal(size=64) * 0.3
        coding = np.tile(coding_center, (20, 1)) + rng.normal(size=(20, 64)) * 0.1

        trace = np.vstack([planning, coding])

        cert = compute_adaptive_certificate(
            trace,
            task_embedding=task_emb,
            window_size=15,
            local_bound_threshold=0.5,
            global_drift_threshold=0.8,
        )

        # Should pass: local stable in coding phase, global still aligned
        assert cert["multi_scale_verdict"] == "PASS"

    def test_agent_goes_off_mission(self):
        """Simulate an agent that starts correct but goes off-mission."""
        rng = np.random.default_rng(42)

        # Task: "Write unit tests"
        task_emb = np.ones(64)

        # Good start: writing tests
        good_start = np.tile(task_emb, (10, 1)) + rng.normal(size=(10, 64)) * 0.1

        # Goes off-mission: starts refactoring unrelated code
        off_mission_center = rng.normal(size=64) * 3.0  # Far from task
        off_mission = np.tile(off_mission_center, (20, 1)) + rng.normal(size=(20, 64)) * 0.1

        trace = np.vstack([good_start, off_mission])

        cert = compute_adaptive_certificate(
            trace,
            task_embedding=task_emb,
            window_size=15,
            local_bound_threshold=0.5,
            global_drift_threshold=0.6,
        )

        # Should fail with goal drift: locally stable but forgot the mission
        assert cert["multi_scale_verdict"] == "FAIL_GOAL_DRIFT"

    def test_agent_becomes_unstable(self):
        """Simulate an agent that becomes erratic/unstable."""
        rng = np.random.default_rng(42)

        task_emb = rng.normal(size=64)

        # Good start
        good_start = np.tile(task_emb, (10, 1)) + rng.normal(size=(10, 64)) * 0.05

        # Becomes unstable: random thrashing
        unstable = rng.normal(size=(20, 64)) * 3.0

        trace = np.vstack([good_start, unstable])

        cert = compute_adaptive_certificate(
            trace,
            task_embedding=task_emb,
            window_size=15,
            local_bound_threshold=0.3,
            global_drift_threshold=0.7,
        )

        # Should fail with instability
        assert cert["multi_scale_verdict"] == "FAIL_INSTABILITY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
