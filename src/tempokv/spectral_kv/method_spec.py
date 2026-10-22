"""Explicit experimental method names. They are never enabled in suite A."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str = "full"
    kind: str = "dct"
    ratio: int = 1
    bias: bool = True
    topk: int | None = None
    window: int | None = None
    sink: int = 1
    exponent: float = 1.0
    taper: str | None = None
    skip: str | None = None

    def to_dict(self):
        return asdict(self)


def parse_method(name: str) -> MethodSpec:
    if name in {"BF16", "FULL"}:
        return MethodSpec(name)
    if "+" in name:
        raise ValueError("suite B does not implement spectral+quantizer combinations; no silent fallback")
    match = re.fullmatch(r"DROP_win([1-9]\d*)(?:_sink(\d+))?", name)
    if match:
        return MethodSpec(name, family="drop", window=int(match[1]), sink=int(match[2] or 1))
    match = re.fullmatch(r"SPECSUM_(dct|avg)_R([1-9]\d*)_B([01])(?:(?:_top([1-9]\d*))|(?:_win([1-9]\d*)))?(_hann)?", name)
    if match:
        if match[6] and match[1] != "dct":
            raise ValueError("Hann taper is defined only for DCT")
        return MethodSpec(name, family="summary", kind=match[1], ratio=int(match[2]), bias=bool(int(match[3])),
                          topk=int(match[4]) if match[4] else None,
                          window=int(match[5]) if match[5] else None, taper="hann" if match[6] else None)
    match = re.fullmatch(r"PYR_(dct|avg)_b(\d+(?:\.\d+)?)", name)
    if match and math.isfinite(float(match[2])) and float(match[2]) >= 0:
        return MethodSpec(name, family="pyramid", kind=match[1], exponent=float(match[2]))
    if name in {"KVSKIP_raw", "KVSKIP_lin"}:
        return MethodSpec(name, family="kvskip", skip=name.split("_")[1])
    raise ValueError(f"unsupported suite B method: {name}")
