import numpy as np
import torch
import tqdm

torch.set_grad_enabled(False)

import triton
import triton.language as tl

from cal_model import CalModel
from utils import fs

from pykeops.torch import LazyTensor


def reference(mic_data, mic_pos, dist_min, k, window_size, output_pos):
    distances = torch.cdist(output_pos, mic_pos)
    distances -= dist_min

    samples_delay = distances * k

    out = torch.zeros(len(output_pos))

    for idx, f in enumerate(range(mic_data.shape[1])):
        phase_delay = torch.exp((1j * np.pi * idx / (window_size // 2)) * samples_delay)
        m = mic_data[:, idx]
        o = m * phase_delay

        # phase_delay_real = torch.cos((np.pi * idx / (window_size // 2)) * samples_delay)
        # phase_delay_imag = torch.sin((np.pi * idx / (window_size // 2)) * samples_delay)
        # assert torch.allclose(phase_delay.real, phase_delay_real)
        # assert torch.allclose(phase_delay.imag, phase_delay_imag)
        # o2 = (m.real * phase_delay_real - m.imag * phase_delay_imag) + \
        #      (m.real * phase_delay_imag + m.imag * phase_delay_real) * 1j
        # assert torch.allclose(o, o2)

        out += o.mean(axis=1).abs() ** 2

    return out


def keops_impl(mic_data, mic_pos, dist_min, k, window_size, output_pos):
    output_pos = LazyTensor(output_pos.view(1, -1, 3))
    mic_pos = LazyTensor(mic_pos.view(-1, 1, 3))

    distances = ((output_pos - mic_pos)**2).sum(dim=2).sqrt()  # shape of (N, M)
    samples_delay = distances * k

    linear_phase_term = LazyTensor((1j * np.pi * torch.arange(mic_data.shape[1], device=mic_data.device) / (window_size // 2)))

    delay_mult = (linear_phase_term * samples_delay).exp()  # shape of (M, N, F)

    out = ((delay_mult * LazyTensor(mic_data, axis=(0))).sum(0).abs() / mic_pos.shape[0]) ** 2

    return out.sum(1)

@triton.heuristics({'num_warps': lambda nargs: 16})
@triton.heuristics({'num_stages': lambda nargs: 64})
@triton.jit
def kernel(
        mic_data_real,  # (M, F)
        mic_data_imag,  # (M, F)
        mic_pos,  # (3, M)
        output_pos,  # (3, N)
        dist_min,  # (N)
        out,  # (N)
        k,  # (1, 1)
        f_start,
        f_end,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        F: tl.constexpr
):
    n_block = tl.program_id(0)
    n_start = n_block * BLOCK_SIZE_N
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)

    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M

    distances = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for dim in range(3):
        mic_pos_dim = tl.load(mic_pos + dim * M + m_offsets, mask=m_mask, other=0.)
        out_pos_dim = tl.load(output_pos + dim * N + n_offsets)
        diff = mic_pos_dim[None, :] - out_pos_dim[:, None]
        distances += diff * diff
    distances = tl.sqrt(distances)
    distances -= tl.load(dist_min + n_offsets)

    k = tl.load(k)

    delay_cycles = distances * k

    out_coherent = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    out_incoherent_energy = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    out_coherent_energy = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)

    for f in range(f_start, f_end):
        data_real = tl.load(mic_data_real + F * m_offsets + f, mask=m_mask, other=0.)[None, :]
        data_imag = tl.load(mic_data_imag + F * m_offsets + f, mask=m_mask, other=0.)[None, :]

        const2 = (np.pi * f * 2.0 / WINDOW_SIZE)

        phase_delay_real = tl.cos(const2 * delay_cycles)
        phase_delay_imag = tl.sin(const2 * delay_cycles)

        out_real = data_real * phase_delay_real - data_imag * phase_delay_imag
        out_imag = data_real * phase_delay_imag + data_imag * phase_delay_real

        out_real = tl.sum(out_real, axis=1) / M
        out_imag = tl.sum(out_imag, axis=1) / M

        coherent_energy = out_real * out_real + out_imag * out_imag
        incoherent_energy = tl.sum(data_real * data_real + data_imag * data_imag, axis=1) / M

        out_coherent += tl.sqrt(coherent_energy)
        out_incoherent_energy += incoherent_energy
        out_coherent_energy += coherent_energy

    tl.atomic_add(out + n_offsets, out_coherent_energy * out_coherent_energy / out_incoherent_energy)


def kernel_fn(mic_data, mic_pos, dist_min, k, window_size, output_pos, f_start=None, f_end=None):
    M = mic_pos.shape[0]
    N = output_pos.shape[0]
    F = mic_data.shape[1]

    f_start = 0 if f_start is None else f_start
    f_end = F if f_end is None else f_end

    out = torch.zeros(N, device="cuda")

    BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_N = 32

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    kernel[grid](
        mic_data.real.contiguous().cuda(),
        mic_data.imag.contiguous().cuda(),
        mic_pos.t().contiguous().cuda(),
        output_pos.t().contiguous().cuda(),
        dist_min.cuda(),
        out,
        k.cuda(),
        f_start,
        f_end,
        window_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        M,
        N,
        F
    )

    return out


def kernel_fn_fast(mic_data_real,  # (M, F)
                   mic_data_imag,  # (M, F)
                   mic_pos,  # (3, M)
                   dist_min,  # (N)
                   k,
                   window_size,
                   output_pos,  # (3, N)
                   f_start=None,
                   f_end=None,
                   out=None):
    assert mic_data_real.shape == mic_data_imag.shape
    assert mic_data_real.is_contiguous()
    assert mic_data_imag.is_contiguous()
    assert mic_pos.is_contiguous()
    assert output_pos.is_contiguous()

    assert mic_data_real.is_cuda
    assert mic_data_imag.is_cuda
    assert mic_pos.is_cuda
    assert output_pos.is_cuda

    assert output_pos.shape[0] == 3
    assert mic_pos.shape[0] == 3

    M = mic_pos.shape[1]
    N = output_pos.shape[1]
    F = mic_data_real.shape[1]

    BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_N = 32

    out = out if out is not None else torch.zeros(N, device="cuda")

    f_start = 0 if f_start is None else f_start
    f_end = F if f_end is None else f_end

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    kernel[grid](
        mic_data_real,
        mic_data_imag,
        mic_pos,
        output_pos,
        dist_min,
        out,
        k,
        f_start,
        f_end,
        window_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        M,
        N,
        F
    )

    return out


def test_correctness():
    x = torch.linspace(-3000, 3000, 16)
    y = torch.linspace(-3000, 3000, 16)
    z = torch.linspace(0, 3000, 16)

    grid = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)

    k = torch.Tensor([fs / 343e3])

    mic_pos = torch.randn(192, 3) * 1000

    time_delay = torch.cdist(grid, mic_pos)
    dist_min = time_delay.min(axis=1, keepdim=True)[0]

    x = torch.randn(192, 768, dtype=torch.cfloat)
    ref = reference(x, mic_pos, dist_min, k, 1024 * 16, grid)
    keops_out = keops_impl(x, mic_pos, dist_min, k, 1024 * 16, grid)
    kernel_out = kernel_fn(x, mic_pos, dist_min, k, 1024 * 16, grid)
    assert torch.allclose(kernel_out, ref.cuda())
    assert torch.allclose(keops_out, ref)

    fast_inputs = (
        x.real.contiguous().cuda(),
        x.imag.contiguous().cuda(),
        mic_pos.t().contiguous().cuda(),
        dist_min.cuda(),
        k.cuda(),
        1024 * 16,
        grid.t().contiguous().cuda(),
    )

    kernel_out_fast = kernel_fn_fast(*fast_inputs, f_start=0, f_end=768, out=torch.zeros_like(kernel_out))
    assert torch.allclose(kernel_out, kernel_out_fast)


def test_performance():
    x = torch.linspace(-3000, 3000, 64)
    y = torch.linspace(-3000, 3000, 64)
    z = torch.linspace(0, 3000, 32)

    grid = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)

    k = torch.Tensor([fs / 343e3])

    mic_pos = torch.randn(128, 3) * 1000

    time_delay = torch.cdist(grid, mic_pos)
    dist_min = time_delay.min(axis=1, keepdim=True)[0]

    x = torch.randn(128, 768, dtype=torch.cfloat)

    fast_inputs = (
        x.real.contiguous().cuda(),
        x.imag.contiguous().cuda(),
        mic_pos.t().contiguous().cuda(),
        dist_min.cuda(),
        k.cuda(),
        1024 * 16,
        grid.t().contiguous().cuda(),
    )

    # keops_inputs = (
    #     x.cuda(),
    #     mic_pos.cuda(),
    #     dist_min.cuda(),
    #     k.cuda(),
    #     1024 * 16,
    #     grid.contiguous().cuda()
    # )

    for i in tqdm.tqdm(range(100)):
        kernel_out_fast = kernel_fn_fast(*fast_inputs, f_start=0, f_end=256).cpu().numpy()
        kernel_out_fast = kernel_fn_fast(*fast_inputs, f_start=256, f_end=512).cpu().numpy()
        kernel_out_fast = kernel_fn_fast(*fast_inputs, f_start=512, f_end=768).cpu().numpy()
        # keops_impl(*keops_inputs)


if __name__ == "__main__":
    test_performance()