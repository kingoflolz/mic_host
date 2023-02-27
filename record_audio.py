import numpy as np
import torch
import tqdm
from scipy.io.wavfile import write
from scipy import signal

from cal_model import CalModel
from utils import decimation, max_offset, clean_bad_mics, fs, clean_bad_mics2
import mic_host


class TimeDomainBeamformer(torch.nn.Module):
    def __init__(self, cal, overlap_size, window_size):
        super().__init__()
        self.cal = cal
        mic_pos = cal.mic_pos
        self.mic_pos = mic_pos
        self.window_size = window_size
        self.overlap_size = overlap_size

        self.buffer = torch.nn.Parameter(
            torch.zeros(
                (mic_pos.shape[0], overlap_size + window_size),
                device=mic_pos.device), requires_grad=False)

    def update(self, data):
        assert data.shape[0] == self.mic_pos.shape[0]
        assert data.shape[1] == self.window_size

        b = torch.concatenate([self.buffer, data], dim=1)

        self.buffer[:] = b[:, -self.window_size-self.overlap_size:]

        ret = self.buffer[:, :self.window_size]
        assert ret.shape == data.shape

        return ret

    def get_audio(self, pos):
        distances = torch.cdist(pos, self.mic_pos)[0]

        distances = distances - distances.max()
        delay = -distances / self.cal.speed_of_sound * fs / 16

        delay_int = delay.long()
        delay_frac = delay - delay_int

        delayed_idx = torch.arange(self.window_size, device=self.mic_pos.device) + \
                      self.overlap_size - delay_int.unsqueeze(1)

        delayed_val = self.buffer[torch.arange(192, device=self.mic_pos.device)[:, None], delayed_idx]
        delayed_val2 = self.buffer[torch.arange(192, device=self.mic_pos.device)[:, None], delayed_idx - 1]

        delayed_val = delayed_val * (1 - delay_frac.unsqueeze(1)) + delayed_val2 * delay_frac.unsqueeze(1)

        assert torch.isfinite(delayed_val).sum() == delayed_val.numel()

        return delayed_val.mean(axis=0)


cal = CalModel(400).cuda()
cal.load_state_dict(torch.load("calibration.pt"))

beamformer = TimeDomainBeamformer(cal, 1024, 1024)

raw = mic_host.RawListener("192.168.1.1:5678", decimation)

chunks_clean = []
chunks_raw = []

pos_noise = torch.nn.Parameter(torch.Tensor([[665.03174, 456.12222, 938.9505]]).cuda())
pos = torch.nn.Parameter(torch.Tensor([[363.2585,  -344.62527, 1607.5024]]).cuda())
opt = torch.optim.Adam([pos, pos_noise], lr=1)

for i in range(100):
    r = raw.chunk(256 * 1024)
    r = np.ascontiguousarray(signal.decimate(r, 16, ftype='iir', zero_phase=True))
    r = torch.Tensor(r).cuda()
    clean_bad_mics2(r)
    raw_audio = beamformer.update(r)

    if i:
        for _ in range(1):
            opt.zero_grad()
            beamformed = beamformer.get_audio(pos)
            loss = -((beamformed - beamformed.mean()) ** 2).mean()

            beamformed_noise = beamformer.get_audio(pos_noise)
            loss += -((beamformed_noise - beamformed_noise.mean()) ** 2).mean()

            loss.backward()
            print(pos.grad.data.detach().cpu().numpy())
            opt.step()
            print(pos.data.detach().cpu().numpy())

        chunks_clean.append(beamformed.detach().cpu().numpy())
        chunks_raw.append(raw_audio[1].detach().cpu().numpy())

        print(i, loss.item(), pos.data.detach().cpu().numpy(), pos_noise.data.detach().cpu().numpy())

f = signal.firwin(31, 500 / (fs / 32), pass_zero=False)

data = np.concatenate(chunks_clean)
data = signal.convolve(data, f, mode='same')
assert np.isfinite(data).all()
data -= data.mean()
factor = np.max(np.abs(data))
scaled = np.int16(data / factor * 32767)
write("out.wav", int(fs / 16), scaled)

data = np.concatenate(chunks_raw)
data = signal.convolve(data, f, mode='same')
assert np.isfinite(data).all()
data -= data.mean()
factor = np.max(np.abs(data))
scaled = np.int16(data / factor * 32767)
write("raw.wav", int(fs / 16), scaled)