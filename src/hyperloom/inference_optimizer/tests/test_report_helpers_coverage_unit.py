# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the report platform line and platform fingerprint."""

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT, VERDICT_KEEP
from hyperloom.orchestrator.actions.executors import report as rp

_OUTPUT_COMPARISON = dict(
    objective=GRADED_OUTPUT,
    reference=100.0,
    candidate=100.0,
    gain_pct=0.0,
    comparable=True,
    degrade_reason="",
    verdict=VERDICT_KEEP,
)


# ---- platform line: absence must be stated, not implied ----
def _summary_without_platform(**extra) -> dict:
    base = {
        "session_id": "s",
        "model_name": "m",
        "model_path": "/m",
        "stop_reason": "target_reached",
        "max_minutes": 60,
        "report_generated_at": "t0",
        "framework": "sglang",
        "current_best": {},
        "baseline_tput": 100.0,
        "cumulative_gain": 0.0,
        "cumulative_gain_validated": 0.0,
        "cumulative_gain_validated_stack_len": 0,
        "optimization_stack_len": 0,
        "crash_count": 0,
        "pruned_families": [],
        "event_counts_by_topic": {},
        "highlights": [],
        "performance_comparison": _OUTPUT_COMPARISON,
    }
    base.update(extra)
    return base


def test_report_says_platform_is_missing_when_the_summary_predates_the_field():
    """A summary with no platform key still gets a line."""
    md = rp._format_md(_summary_without_platform())
    assert "- platform       : not recorded" in md


def test_report_gives_the_reason_the_platform_is_missing_when_it_has_one():
    md = rp._format_md(_summary_without_platform(platform={"status": "unavailable", "reason": "no host CPU sysfs"}))
    assert "not recorded — no host CPU sysfs" in md


def test_report_names_the_board_the_run_was_on():
    identity = {
        "product": "w7900",
        "marketing_name": "AMD Radeon Pro W7900 Dual Slot",
        "pci_device_id": "1002:744a",
        "gfx_arch": "gfx1100",
        "visible_gpus": 1,
    }
    platform = {
        "status": "ok",
        "host": "gpu-host-01",
        "gpu": {"host_count": 4, "gfx_arch": "gfx1100", "amdgpu_driver": "6.16.13", "identity": identity},
    }
    md = rp._format_md(_summary_without_platform(platform=platform))
    assert (
        "- accelerators   : 4× gfx1100 on the host, amdgpu 6.16.13; "
        "run on w7900, AMD Radeon Pro W7900 Dual Slot, 1002:744a"
    ) in md


def test_report_names_only_what_the_identity_recorded():
    platform = {
        "status": "ok",
        "host": "h",
        "gpu": {"host_count": 8, "gfx_arch": "gfx942", "identity": {"product": "mi300x", "marketing_name": None}},
    }
    assert "- accelerators   : 8× gfx942 on the host, amdgpu unknown; run on mi300x\n" in rp._format_md(
        _summary_without_platform(platform=platform)
    )


@pytest.mark.parametrize(("kernel_enabled", "shown"), [(False, True), (True, False), (None, False)])
def test_report_says_when_the_kernel_phase_was_off(kernel_enabled, shown):
    extra = {} if kernel_enabled is None else {"kernel_enabled": kernel_enabled}
    md = rp._format_md(_summary_without_platform(**extra))
    assert ("- kernel phase   : disabled (--no-kernel)" in md) is shown


def test_platform_fingerprint_propagates_a_node_scope_failure(monkeypatch):
    import pytest

    def _boom():
        raise RuntimeError("node scope unreadable")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", _boom)

    with pytest.raises(RuntimeError, match="node scope unreadable"):
        rp._platform_fingerprint()
