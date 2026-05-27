# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# CUDA-graph-safe NaN trap for diagnosing where bad values first appear.
#
# Enable: TRTLLM_NAN_TRAP=1
# Tune depth: TRTLLM_NAN_TRAP_DEPTH=2 (default; hooks individual decoder layers)
# Focus: TRTLLM_NAN_TRAP_FOCUS=model.layers.0,model.layers.1 (hook ALL children)
# Sub-modules: TRTLLM_NAN_TRAP_SUB=self_attn,mlp (hook these children of depth layers)
# ModuleList/Sequential containers don't count toward depth.
#
# Hooks use device-side flag tensor + armed gate (bypasses warmup).
# Cost: 2 kernels per hooked layer (isnan+any, then logical_or_ into flags).

from __future__ import annotations

import os
from typing import List, Optional

import torch

from tensorrt_llm.logger import logger

_CHECKABLE_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
              "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _CHECKABLE_DTYPES.add(_dt)


def _walk_floating_tensors(obj):
    """Yield all checkable floating tensors from a module output."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype in _CHECKABLE_DTYPES:
            yield obj
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _walk_floating_tensors(x)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from _walk_floating_tensors(x)
        return
    if hasattr(obj, "__dict__"):
        for x in vars(obj).values():
            yield from _walk_floating_tensors(x)


class NanTrap:

    def __init__(self, names: List[str], device: torch.device):
        self.names = names
        self.flags = torch.zeros(len(names), dtype=torch.bool, device=device)
        self._armed = torch.zeros(1, dtype=torch.bool, device=device)
        self._step = 0

    def arm(self):
        self._armed.fill_(True)

    def check_and_log(self, rank: int) -> None:
        self._step += 1
        if not self._armed.item():
            return
        try:
            flags_host = self.flags.to("cpu", non_blocking=False)
        except RuntimeError:
            return
        bad = flags_host.nonzero(as_tuple=False).flatten().tolist()
        if not bad:
            return
        first = self.names[bad[0]]
        all_names = [self.names[i] for i in bad[:16]]
        logger.error(
            f"[NaN-TRAP] rank={rank} step={self._step} first_nan_module={first!r} "
            f"num_flagged={len(bad)} flagged_modules={all_names}")
        self.flags.zero_()


def maybe_attach_nan_trap(model: torch.nn.Module) -> Optional[NanTrap]:
    if os.environ.get("TRTLLM_NAN_TRAP", "0") != "1":
        return None
    max_depth = int(os.environ.get("TRTLLM_NAN_TRAP_DEPTH", "2"))
    focus_raw = os.environ.get("TRTLLM_NAN_TRAP_FOCUS", "")
    focus_prefixes = (
        [p.strip() for p in focus_raw.split(",") if p.strip()]
        if focus_raw else None
    )
    sub_raw = os.environ.get("TRTLLM_NAN_TRAP_SUB", "")
    sub_names = (
        [s.strip() for s in sub_raw.split(",") if s.strip()]
        if sub_raw else None
    )

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.device("cuda"),
    )

    container_prefixes: set = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            container_prefixes.add(name)

    names: List[str] = []
    modules = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        if focus_prefixes:
            in_focus = any(
                name == fp or name.startswith(fp + ".")
                for fp in focus_prefixes
            )
            if in_focus:
                names.append(name)
                modules.append(mod)
                continue
        parts = name.split(".")
        depth = sum(1 for i, _ in enumerate(parts)
                    if ".".join(parts[:i + 1]) not in container_prefixes)
        if depth > max_depth:
            continue
        names.append(name)
        modules.append(mod)

    if sub_names and not focus_prefixes:
        hooked_set = set(names)
        for name, mod in model.named_modules():
            if name == "" or isinstance(
                    mod, (torch.nn.ModuleList, torch.nn.Sequential)):
                continue
            if name in hooked_set:
                continue
            tail = name.rsplit(".", 1)[-1]
            if tail not in sub_names:
                continue
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if parent in hooked_set:
                names.append(name)
                modules.append(mod)

    trap = NanTrap(names, device)
    for i, mod in enumerate(modules):

        def make_hook(slot):

            def hook(_m, _inp, out):
                try:
                    last = None
                    for t in _walk_floating_tensors(out):
                        last = t
                    if last is not None:
                        has_nan = torch.any(torch.isnan(last))
                        trap.flags[slot].logical_or_(
                            trap._armed.squeeze() & has_nan)
                except (NotImplementedError, RuntimeError):
                    pass

            return hook

        mod.register_forward_hook(make_hook(i))

    logger.info(
        f"[NaN-TRAP] attached {len(names)} hooks (max_depth={max_depth})"
        f"{f' focus={focus_prefixes}' if focus_prefixes else ''}"
        f"{f' sub={sub_names}' if sub_names else ''}")
    return trap
