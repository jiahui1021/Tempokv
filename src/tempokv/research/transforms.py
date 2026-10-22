"""Pre-RoPE spectral diagnostics with explicit dense-grid and temporal semantics."""
from __future__ import annotations
import torch
from tempokv.spectral.transforms import grid_shape, spatial_layout, spatial_blocks, spatial_fft, temporal_fft
from tempokv.spectral.spectrum import temporal_energy
from tempokv.spectral_kv.projection import dct_matrix, make_projection, compress_grid, reconstruct_grid


def radial_mask(h,w,cutoff,*,device=None):
    fy=torch.fft.fftfreq(h,device=device,dtype=torch.float64)
    fx=torch.fft.fftfreq(w,device=device,dtype=torch.float64)
    return fy[:,None].square()+fx[None,:].square()<=float(cutoff)**2+8*torch.finfo(torch.float64).eps


def spatial_energy(tensor,cutoffs,eps=1e-12):
    power=spatial_fft(tensor).abs().square().sum(dim=-1)
    total=power.sum(dim=(2,3));zero=total<=eps;ratios={}
    for cutoff in cutoffs:
        mask=radial_mask(tensor.shape[2],tensor.shape[3],cutoff,device=tensor.device)
        low=(power*mask[None,None,:,:,None]).sum(dim=(2,3))
        ratios[cutoff]=(low/(total+eps)).masked_fill(zero,float("nan"))
    return ratios,total,zero


def dct2(x):
    if x.ndim != 5: raise ValueError("DCT input must be [F,H,W,heads,D]")
    dh, dw = dct_matrix(x.shape[1]).to(x.device), dct_matrix(x.shape[2]).to(x.device)
    return torch.einsum("ah,fhwnd,bw->fabnd", dh, x.float(), dw)


def idct2(coefficients):
    if coefficients.ndim != 5: raise ValueError("inverse DCT input must be [F,H,W,heads,D]")
    dh, dw = dct_matrix(coefficients.shape[1]).to(coefficients.device), dct_matrix(coefficients.shape[2]).to(coefficients.device)
    return torch.einsum("ah,fabnd,bw->fhwnd", dh, coefficients.float(), dw)
