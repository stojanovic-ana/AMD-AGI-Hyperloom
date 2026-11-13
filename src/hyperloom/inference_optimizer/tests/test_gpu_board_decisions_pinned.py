# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every per-board decision, pinned to its value, so a change to the board tables shows exactly what it moves."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS, resolve_compute_peak_provenance
from hyperloom.orchestrator.actions.executors._server_lifecycle import MAGPIE_BUILTIN_SCRIPTS

_BOARDS = [
    # product, runner, targeted build validated, bypass peak convention
    ("mi300x", "mi300x", True, "achievable"),
    ("mi308x", "mi300x", True, "vendor"),
    ("mi325x", "mi300x", True, "achievable"),
    ("mi350x", "mi355x", True, "vendor"),
    ("mi355x", "mi355x", True, "vendor"),
    ("r9700", "gfx12", False, "vendor"),
]


@pytest.mark.parametrize(("product", "runner", "build_validated", "convention"), _BOARDS)
def test_board_decisions(product, runner, build_validated, convention):
    assert gpu_types.gpu_runner_type(product) == runner
    assert gpu_types.targeted_build_validated(product) is build_validated
    assert resolve_compute_peak_provenance(product, "bf16")["compute_peak_convention"] == convention


def test_radeon8060s_has_its_own_runner_and_no_framework_restriction():
    assert gpu_types.gpu_runner_type("radeon8060s") == "radeon8060s"
    assert gpu_types.targeted_build_validated("radeon8060s") is True
    assert gpu_types._runner_framework_error("radeon8060s", "sglang") is None


def test_w7900_runs_the_vllm_only_gfx11_runner_with_published_specs():
    assert gpu_types.gpu_runner_type("w7900") == "gfx11"
    assert gpu_types.targeted_build_validated("w7900") is False
    assert gpu_types._runner_framework_error("w7900", "vllm") is None
    assert gpu_types._runner_framework_error("w7900", "sglang") == (
        "Magpie runner 'gfx11' is currently validated only for vllm; framework 'sglang' has no validated gfx11 runner"
    )
    assert "w7900" in HW_SPECS


@pytest.mark.parametrize("product", [*(board[0] for board in _BOARDS), "radeon8060s", "w7900"])
def test_only_w7900_leaves_its_arch_spec_unmeasured(product):
    assert gpu_types.arch_spec_measurable(product) is (product != "w7900")


def test_server_reuse_scripts():
    assert MAGPIE_BUILTIN_SCRIPTS == {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "vllm_gfx11.sh",
        "vllm_gfx12.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }


@pytest.mark.parametrize("product", ["mi300x", "mi308x", "mi325x", "mi350x", "mi355x"])
@pytest.mark.parametrize("framework", ["vllm", "sglang", "atom", "xdit"])
def test_instinct_boards_leave_the_framework_unconstrained(product, framework):
    assert gpu_types._runner_framework_error(product, framework) is None


@pytest.mark.parametrize("framework", ["sglang", "atom", "xdit"])
def test_the_gfx12_runner_is_validated_for_vllm_only(framework):
    assert gpu_types._runner_framework_error("r9700", "vllm") is None
    assert gpu_types._runner_framework_error("r9700", framework) == (
        "Magpie runner 'gfx12' is currently validated only for vllm; "
        f"framework {framework!r} has no validated gfx12 runner"
    )


def test_only_r9700_and_w7900_pin_their_geak_identity():
    assert gpu_types.geak_identity_pin("r9700") == {"expected_target": "r9700", "expected_gfx": "gfx1201"}
    assert gpu_types.geak_identity_pin("w7900") == {"expected_target": "w7900", "expected_gfx": "gfx1100"}
    for product in ("mi300x", "mi308x", "mi325x", "mi350x", "mi355x", "radeon8060s", "unknown_gpu", ""):
        assert gpu_types.geak_identity_pin(product) == {}, product


@pytest.mark.parametrize(
    ("text", "product"),
    [
        ("Card Series: AMD Instinct MI300X OAM", "mi300x"),
        ("Card Series: AMD Instinct MI300X VF", "mi300x"),
        ("Card Series: AMD Instinct MI308X OAM", "mi308x"),
        ("Card Series: AMD Instinct MI325X OAM", "mi325x"),
        ("Card Series: AMD Instinct MI350X OAM", "mi350x"),
        ("Card Series: AMD Instinct MI355X OAM", "mi355x"),
        ("AMD_INSTINCT_MI300X", "mi300x"),
        ("Card Series: AMD Radeon AI PRO R9700", "r9700"),
        ("card series: amd radeon ai pro r9700", "r9700"),
        ("Card Series: AMD Instinct MI300XL", None),
        ("Card Series: AMD Radeon AI PRO R9700S", None),
        ("Card Series: AMD Radeon AI PRO XR9700", None),
        ("Card Series: AMD Radeon 8060S Graphics", "radeon8060s"),
        ("gfx1151", None),
        ("gfx942:sramecc+:xnack-", "mi300x"),
        ("gfx950:sramecc+:xnack-", "mi355x"),
        ("gfx1201", None),
        ("gfx1100", None),
        ("nothing here", None),
        ("", None),
        (None, None),
        # W7900: only one of its two exact nameplates, in any case, from rocm-smi or rocminfo.
        ("Card Series: AMD Radeon Pro W7900", "w7900"),
        ("Card Series: AMD Radeon Pro W7900 Dual Slot", "w7900"),
        ("GPU[0]\t\t: Card Series: \t\tAMD Radeon Pro W7900 Dual Slot", "w7900"),
        ("Marketing Name:          AMD Radeon Pro W7900", "w7900"),
        ("Marketing Name:          AMD Radeon Pro W7900 Dual Slot     ", "w7900"),
        ("Card Series: AMD RADEON PRO W7900", "w7900"),
        ("Card Series: amd radeon pro w7900 dual slot", "w7900"),
        ("AMD Radeon PRO W7900 Dual Slot", "w7900"),
        ("Card Series: AMD Radeon PRO W7800", None),
        ("Card Series: AMD Radeon PRO W7800 48GB", None),
        ("Card Series: AMD Radeon RX 7900 XTX", None),
        ("Card Series: AMD Radeon RX 7900 XT", None),
        ("Card Series: AMD Radeon RX 7900 GRE", None),
        ("Card Series: W7900", None),
        ("Card Series: Radeon Pro W7900", None),
        ("Card Series: AMD Radeon Pro W7900 Triple Slot", None),
        ("w7900", None),
        ("Card Series: N/A", None),
    ],
)
def test_probe_output_resolves_to_the_same_board(text, product):
    """A longer product name that contains a tag (``MI300XL``, ``R9700S``) is a different product."""
    assert gpu_types.parse_gpu_type(text) == product


def test_the_validation_hosts_rocm_smi_output_names_w7900():
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "rocm_smi_showproductname_w7900.txt"
    assert gpu_types.parse_gpu_type(fixture.read_text(encoding="utf-8")) == "w7900"
