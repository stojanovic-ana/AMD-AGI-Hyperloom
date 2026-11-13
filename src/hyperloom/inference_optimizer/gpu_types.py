# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_common.gpu_identity import (
    AMD_GPU_BOARDS,
    GpuAgent,
    GpuBoard,
    board_named_in,
    board_of_agent,
    gpu_board,
    normalise_device_id,
)


@dataclass(frozen=True)
class MagpieRunner:
    """A Magpie benchmark-script family, ``{framework}_{name}.sh``."""

    name: str
    #: Frameworks whose ``{framework}_{name}.sh`` supports Magpie's server_lifecycle reuse protocol.
    reuse_frameworks: frozenset[str]
    #: Frameworks validated on this runner; ``None`` leaves the framework unconstrained.
    validated_frameworks: frozenset[str] | None = None


@dataclass(frozen=True)
class BoardPolicy:
    """What Hyperloom does on one board of ``AMD_GPU_BOARDS``."""

    runner: MagpieRunner
    #: Whether the targeted-build (compiled / AITER) path has been validated on this board.
    targeted_build_validated: bool = True
    #: Whether the GEAK handoff names the product and ISA instead of leaving them to GEAK's device probe.
    geak_identity_pinned: bool = False
    #: Whether a bare arch (``gcnArchName``, rocminfo) stands for this board; other boards need their product name.
    names_its_arch: bool = False
    #: Whether trace analysis may measure an arch spec on the card when TraceLens bundles none. Without one, the
    #: roofline ceilings are recorded as unavailable.
    measures_arch_spec: bool = True


_VLLM_ONLY = frozenset({"vllm"})
_INSTINCT_FRAMEWORKS = _VLLM_ONLY | {"sglang", "atom"}
_MI300X_RUNNER = MagpieRunner("mi300x", _INSTINCT_FRAMEWORKS)
_MI355X_RUNNER = MagpieRunner("mi355x", _INSTINCT_FRAMEWORKS)
_GFX12_RUNNER = MagpieRunner("gfx12", _VLLM_ONLY, validated_frameworks=_VLLM_ONLY)
_GFX11_RUNNER = MagpieRunner("gfx11", _VLLM_ONLY, validated_frameworks=_VLLM_ONLY)

#: Keyed by ``AMD_GPU_BOARDS``: every board has exactly one policy row.
_BOARD_POLICIES: dict[str, BoardPolicy] = {
    "mi300x": BoardPolicy(_MI300X_RUNNER, names_its_arch=True),
    "mi308x": BoardPolicy(_MI300X_RUNNER),
    "mi325x": BoardPolicy(_MI300X_RUNNER),
    # The same gfx950 die as mi355x, so it runs mi355x's Magpie scripts; gfx950 alone still names mi355x.
    "mi350x": BoardPolicy(_MI355X_RUNNER),
    "mi355x": BoardPolicy(_MI355X_RUNNER, names_its_arch=True),
    "r9700": BoardPolicy(_GFX12_RUNNER, targeted_build_validated=False, geak_identity_pinned=True),
    "radeon8060s": BoardPolicy(MagpieRunner("radeon8060s", frozenset())),
    "w7900": BoardPolicy(
        _GFX11_RUNNER, targeted_build_validated=False, geak_identity_pinned=True, measures_arch_spec=False
    ),
}

_RUNNERS: dict[str, MagpieRunner] = {policy.runner.name: policy.runner for policy in _BOARD_POLICIES.values()}

#: A bare arch resolves only to a board marked ``names_its_arch`` (gfx942 to mi300x, gfx950 to mi355x); gfx12,
#: gfx1151 and gfx1100 cover several products, so those resolve through the product name.
_GFX_REFERENCE_BOARDS: dict[str, str] = {
    AMD_GPU_BOARDS[product].gfx: product for product, policy in _BOARD_POLICIES.items() if policy.names_its_arch
}


def _board_policy(gpu_type: str | None) -> BoardPolicy | None:
    return _BOARD_POLICIES.get(str(gpu_type or "").strip().lower())


def gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    policy = _board_policy(gpu_type)
    return policy.runner.name if policy else str(gpu_type or "").strip().lower()


def magpie_reuse_scripts() -> frozenset[str]:
    """The Magpie benchmark scripts that support the server_lifecycle reuse protocol."""
    return frozenset(
        f"{framework}_{runner.name}.sh" for runner in _RUNNERS.values() for framework in runner.reuse_frameworks
    )


def targeted_build_validated(gpu_type: str) -> bool:
    """Whether the targeted-build (compiled / AITER) path has been validated on this board."""
    policy = _board_policy(gpu_type)
    return policy.targeted_build_validated if policy else True


def arch_spec_measurable(gpu_type: str | None) -> bool:
    """Whether trace analysis may measure an arch spec on this board when TraceLens bundles none."""
    policy = _board_policy(gpu_type)
    return policy.measures_arch_spec if policy else True


def geak_identity_pin(gpu_type: str) -> dict[str, str]:
    """The ``expected_target`` / ``expected_gfx`` pair a GEAK handoff carries for this board, if any."""
    policy = _board_policy(gpu_type)
    board = gpu_board(gpu_type)
    if not (policy and board and policy.geak_identity_pinned):
        return {}
    return {"expected_target": board.product, "expected_gfx": board.gfx}


def geak_reads_the_nameplate(gpu_type: str | None) -> bool:
    """Whether GEAK, handed this board's pin, probes the card itself and refuses one its identity helper cannot name.

    True for a pinned board named only by its nameplate, which no ISA stands for.
    """
    board = gpu_board(gpu_type)
    return bool(board and board.marketing_names and geak_identity_pin(board.product))


def reference_gpu_type_for_gfx(text: str | None) -> str | None:
    """Return the board a bare gfx arch in ``text`` (``gcnArchName``, rocminfo) stands for, or ``None``."""
    lower = str(text or "").lower()
    return next((product for gfx, product in _GFX_REFERENCE_BOARDS.items() if gfx in lower), None)


def parse_gpu_type(text: str | None) -> str | None:
    """Map rocm-smi / gcnArchName output, possibly mixed into other log text, to a supported board name."""
    board = board_named_in(text)
    return board.product if board else reference_gpu_type_for_gfx(text)


def _runner_framework_error(gpu_type: str, framework: str) -> str | None:
    """Return why a constrained runner/framework pair is unavailable."""
    runner_name = gpu_runner_type(gpu_type)
    runner = _RUNNERS.get(runner_name)
    framework = str(framework or "").strip().lower()
    supported = runner.validated_frameworks if runner else None
    if not framework or not supported or framework in supported:
        return None
    return (
        f"Magpie runner {runner_name!r} is currently validated only for "
        f"{', '.join(sorted(supported))}; framework {framework!r} has no validated {runner_name} runner"
    )


#: ISAs that enable nothing without the board's exact nameplate: each is shared by boards Hyperloom has not validated.
_PRODUCT_REQUIRED_GFX = frozenset({"gfx1100", "gfx1101", "gfx1102"})

_MASK_HINT = "If the host has other GPUs, set ROCR_VISIBLE_DEVICES to the GPUs this session should use."


def _visible_gpu(agent: GpuAgent) -> str:
    """``the visible GPU``, with the nameplate or else the PCI ID rocminfo printed for it."""
    label = repr(agent.marketing_name) if agent.marketing_name else agent.pci_device_id
    return f"the visible GPU {label}" if label else "the visible GPU"


def _pci_list(board: GpuBoard) -> str:
    return " or ".join(f"1002:{device_id}" for device_id in sorted(board.device_ids))


def _bare_isa_error(requested: str, board: GpuBoard | None, agent: GpuAgent) -> str:
    gfx = agent.gfx
    supported = ", ".join(
        f"{b.product} ({', '.join(b.marketing_names)})" for b in AMD_GPU_BOARDS.values() if b.gfx == gfx
    )
    named = f"{requested} dispatches to {board.gfx}, but" if board else "no supported product was identified;"
    return (
        f"{named} {_visible_gpu(agent)} reports {gfx}, and a bare {gfx} is not product evidence. "
        + (f"Supported {gfx} products: {supported}." if supported else f"No {gfx} product is validated.")
        + f" {_MASK_HINT}"
    )


def _nameplate_board_error(named: str, board: GpuBoard, agent: GpuAgent) -> str | None:
    """Why ``agent`` is not ``board``: each agent needs its ISA and its nameplate or PCI ID, and neither may differ."""
    if agent.gfx != board.gfx:
        error = f"{named} dispatches to {board.gfx}, but {_visible_gpu(agent)} reports {agent.gfx}"
    elif agent.marketing_name and not board.named_in(agent.marketing_name):
        error = f"{named} is {_nameplate_list(board.product)}, but the visible GPU is {agent.marketing_name!r}"
    elif agent.chip_id and board.device_ids and normalise_device_id(agent.chip_id) not in board.device_ids:
        gpu = f"the visible GPU {agent.marketing_name!r}" if agent.marketing_name else "the visible GPU"
        error = f"{named} is PCI {_pci_list(board)}, but {gpu} has PCI ID {agent.pci_device_id}"
    elif not (agent.marketing_name or agent.chip_id):
        error = (
            f"{named} is identified by its nameplate or PCI ID, and rocminfo prints neither for the visible "
            f"{agent.gfx} GPU"
        )
    else:
        return None
    return f"{error}. {_MASK_HINT}"


def probed_gpu_type(agents: Sequence[GpuAgent] | None) -> str:
    """The board the first rocminfo agent's nameplate or PCI ID names, or ``""``."""
    for agent in agents or ():
        board = board_of_agent(agent)
        if board:
            return board.product
    return ""


def product_launch_error(
    gpu_type: str | None,
    agents: Sequence[GpuAgent] | None,
    *,
    framework: str,
    resumed: bool = False,
    remote: bool = False,
) -> str | None:
    """Why this launch cannot run on the board, or ``None``; checked before any install, LLM call or session dir.

    ``gpu_type`` is the flag (or env) value, or with ``resumed`` the one the resumed session was created with;
    ``agents`` is what rocminfo shows the session, or ``None`` when it cannot run. A board named only by its
    nameplate or PCI ID is accepted only when every agent shows its ISA and its nameplate or PCI ID, so a local
    launch that rocminfo shows no GPU for is refused. Other boards, and a ``remote`` launch (the GPUs are on other
    nodes), check only what rocminfo shows.
    """
    requested = str(gpu_type or "").strip().lower()
    probed = probed_gpu_type(agents)
    conflict = gpu_type_conflict(requested, probed, resumed=resumed)
    if conflict:
        return conflict
    product = requested or probed
    named = f"the session's GPU type {product}" if resumed else f"--gpu-type {product}"
    board = gpu_board(product)
    nameplate_board = board if board and board.marketing_names else None
    if nameplate_board and not remote and not agents:
        if agents is None:
            seen, fix = "could not run", "Put rocminfo on PATH and give"
        else:
            seen, fix = "lists no GPU", "Check that ROCR_VISIBLE_DEVICES does not hide the card, and give"
        return (
            f"{named} is identified by its nameplate or PCI ID, but rocminfo {seen}, so the card is unconfirmed. "
            f"{fix} the session the card's /dev/kfd and /dev/dri nodes."
        )
    for agent in agents or ():
        if agent.gfx in _PRODUCT_REQUIRED_GFX and not (board and board.gfx == agent.gfx):
            return _bare_isa_error(named, board, agent)
        error = _nameplate_board_error(named, nameplate_board, agent) if nameplate_board else None
        if error:
            return error
    return _runner_framework_error(product, framework) if board else None


def require_runner_framework(gpu_type: str, framework: str) -> None:
    """Exit at launch when the selected runner cannot serve the framework."""
    unsupported = _runner_framework_error(gpu_type, framework)
    if unsupported:
        print(f"ERROR: {unsupported}", file=sys.stderr)
        sys.exit(2)


def warn_family_runner(gpu_type: str, framework: str) -> None:
    """Say when a product launches through its architecture family's Magpie runner."""
    runner = gpu_runner_type(gpu_type)
    if gpu_type and runner != gpu_type:
        print(
            f"WARN: {gpu_type} uses {runner} as Magpie runner_type (architecture-family runner; "
            f"Magpie has no per-SKU {framework}_{gpu_type}.sh)",
            file=sys.stderr,
        )


def gpu_type_conflict(user_specified: str | None, probed: str | None, *, resumed: bool = False) -> str | None:
    """Why a GPU type the probe contradicts is refused when either side is a board named only by its nameplate.

    Every other disagreement keeps the probe-wins rule of :func:`_resolve_gpu_type`. With ``resumed``, the GPU type
    is the one the resumed session was created with, which no flag changes.
    """
    user = str(user_specified or "").strip().lower()
    probe = str(probed or "").strip().lower()
    if not (user and probe) or user == probe:
        return None
    if not any(board and board.marketing_names for board in (gpu_board(user), gpu_board(probe))):
        return None
    if resumed:
        return (
            f"this session was created for {user}, but the probed board is {probe}; a "
            f"{_nameplate_list(user, probe)} session must resume on the board it was created on."
        )
    return (
        f"--gpu-type {user} disagrees with the probed {probe}; a {_nameplate_list(user, probe)} launch must match the "
        "board it runs on. Pass the --gpu-type the probe reports, or run on the board you named."
    )


def require_gpu_type_agreement(user_specified: str | None, probed: str | None) -> None:
    """Exit at launch when :func:`gpu_type_conflict` refuses the GPU type the probe contradicts."""
    conflict = gpu_type_conflict(user_specified, probed)
    if conflict:
        print(f"ERROR: {conflict}", file=sys.stderr)
        sys.exit(2)


def _nameplate_list(*gpu_types: str) -> str:
    names = [
        f"{board.product} ({', '.join(board.marketing_names)})"
        for board in (gpu_board(gpu_type) for gpu_type in gpu_types)
        if board and board.marketing_names
    ]
    return " / ".join(names)


def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe."""
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"Magpie runner_type + KB recipe rows must match the actual "
            f"hardware to keep baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings


def _autodetect_gpu_type(agents: Sequence[GpuAgent] | None = None) -> str | None:
    """Return a canonical supported board name, or ``None`` if undetectable.

    ``agents`` is what the launch's rocminfo run showed (``cli.launch_gpu.launch_gpu_probe``); this runs no rocminfo
    itself.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        detected = parse_gpu_type(out)
        if detected:
            return detected
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the rocminfo agents and the torch probe below
        # (autodetect is best-effort).
        pass
    # rocm-smi prints "Card Series: N/A" when libdrm refuses it, and some boards are named only by that nameplate.
    detected = probed_gpu_type(agents)
    if detected:
        return detected
    try:
        import torch

        return parse_gpu_type(str(torch.cuda.get_device_properties(0).gcnArchName))
    except (ImportError, OSError, RuntimeError, AssertionError, AttributeError):
        return None


def _resolve_amd_gpu_type(explicit: str | None = None, *, probe: bool = True) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown; ``probe=False`` reads the flag and env only."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        return explicit_norm if explicit_norm in AMD_GPU_BOARDS else None
    env_norm = (os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")).strip().lower()
    if env_norm:
        return env_norm if env_norm in AMD_GPU_BOARDS else None
    if not probe:
        return None
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in AMD_GPU_BOARDS else None


def amd_gpu_board(gpu_type: str | None = None) -> GpuBoard | None:
    """Return the board of an explicit GPU type, ``$TARGET_GPU_TYPE`` / ``$GPU_TYPE``, or the detected GPU."""
    return gpu_board(_resolve_amd_gpu_type(gpu_type))
