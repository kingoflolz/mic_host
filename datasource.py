import itertools

from cal_model import CalModel
from utils import fs, decimation, clean_bad_mics2, locations
import queue
import threading

import mic_host
import numpy as np
import torch

from kernel import kernel_fn_fast


class DataSource:
    def __init__(self, shape, ax_lims, window_size=16 * 1024):
        self.window_size = window_size

        self.shape = shape

        x = torch.linspace(*ax_lims[0], self.shape[0])
        y = torch.linspace(*ax_lims[1], self.shape[1])
        z = torch.linspace(*ax_lims[2], self.shape[2])

        grid = torch.stack(torch.meshgrid(x, y, z), dim=-1).permute([1, 0, 2, 3]).reshape(-1, 3).cuda()

        cal = CalModel(200).cuda()
        cal.load_state_dict(torch.load("calibration.pt"))
        self.cal = cal

        self.k = torch.Tensor([fs / cal.speed_of_sound]).cuda()
        self.freqs = np.fft.rfftfreq(self.window_size - 1, 1 / fs)

        self.freq_ranges = [
            (500,  1700),
            (1700, 3000),
            (3000, 8000),
            # (500,  10000),
        ]

        self.freq_idx_ranges = np.searchsorted(self.freqs, self.freq_ranges)

        rings = [
            [i for i in range(192) if i % 8 == j] for j in range(8)
        ]

        self.mic_indexes = [
            list(itertools.chain(*rings[3:8])) + rings[1][::6] + rings[2][3::6],
            list(itertools.chain(*rings[2:7])) + rings[1][::6] + rings[7][3::6],
            list(itertools.chain(*rings[0:5])) + rings[5][::6] + rings[6][3::6],
            # list(itertools.chain(*rings))
        ]

        def mic_set_cache(mic_pos):
            out = {}
            dist = torch.cdist(grid, mic_pos)
            out["dist_min"] = dist.min(axis=1, keepdim=True)[0].cuda()
            out["mic_pos"] = mic_pos.t().contiguous().cuda()
            return out

        self.mic_set_cache = [mic_set_cache(cal.mic_pos[mic_index]) for mic_index in self.mic_indexes]

        self.grid = grid.t().contiguous().cuda()

        self.raw = mic_host.RawListener("192.168.1.1:5678", decimation)
        self.corr = mic_host.Correlator(self.window_size)
        self.raw.chunk(self.window_size * decimation)

        self.input_queue = queue.Queue(2)
        self.push_queue = queue.Queue(2)

        def input_thread():
            while True:
                chunk = self.raw.chunk(self.window_size * decimation)
                self.push_queue.put(chunk)

        def push_thread():
            while True:
                chunk = self.push_queue.get()
                chunk = clean_bad_mics2(chunk)
                rfft = self.corr.rfft(chunk)
                rfft = torch.from_numpy(rfft)[:, :self.freq_idx_ranges.max()].cuda()

                blocks = [rfft[mic_index] for mic_index in self.mic_indexes]
                blocks = [(b.real.contiguous(), b.imag.contiguous()) for b in blocks]
                self.input_queue.put(blocks)

        self.input_t = threading.Thread(target=input_thread, daemon=True).start()
        self.push_t = threading.Thread(target=push_thread, daemon=True).start()

    def post_process(self, data):
        raise NotImplemented

    def get(self):
        data = self.input_queue.get()
        outs = []
        for ((r, i), cache, (f_start, f_stop)) in zip(data, self.mic_set_cache, self.freq_idx_ranges):
            outs.append(kernel_fn_fast(r,
                                       i,
                                       cache["mic_pos"],
                                       cache["dist_min"],
                                       self.k,
                                       self.window_size,
                                       self.grid,
                                       f_start=int(f_start),
                                       f_end=int(f_stop),))

        out = sum(outs)
        out = out.reshape(*self.shape)

        return self.post_process(out)

    def start(self):
        def data_thread(self, queue_out):
            torch.set_grad_enabled(False)
            while True:
                queue_out.put(self.get())
                print("produced data")

        def sync_thread(queue_in, queue_out):
            torch.set_grad_enabled(False)
            while True:
                queue_out.put(queue_in.get().cpu().numpy())
                print("synced data")

        data_queue = queue.Queue(2)
        out_queue = queue.Queue(2)
        self.read_thread = threading.Thread(target=data_thread, args=(self, data_queue,), daemon=True).start()
        self.sync_thread = threading.Thread(target=sync_thread, args=(data_queue, out_queue,), daemon=True).start()
        return out_queue

    def plot_arrays(self):
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        # Plot the data on each subplot
        axs[0, 0].scatter(locations[self.mic_indexes[0], 0], locations[self.mic_indexes[0], 1])
        axs[0, 1].scatter(locations[self.mic_indexes[1], 0], locations[self.mic_indexes[1], 1])
        axs[1, 0].scatter(locations[self.mic_indexes[2], 0], locations[self.mic_indexes[2], 1])
        axs[1, 1].scatter(locations[:, 0], locations[:, 1])

        axs[0, 0].set_title(f'Subarray {self.freq_ranges[0]} hz')
        axs[0, 1].set_title(f'Subarray {self.freq_ranges[1]} hz')
        axs[1, 0].set_title(f'Subarray {self.freq_ranges[2]} hz')
        axs[1, 1].set_title('Full array')

        plt.show()


class DataSource3D(DataSource):
    def __init__(self):
        super().__init__(shape=(64, 64, 64), ax_lims=((-1600, 1600), (-1600, 1600), (0, 3200)))

    def post_process(self, data):
        d_min = data.min()
        data -= d_min
        d_max = data.max()
        data /= d_max
        print(f"snr: {d_max / d_min}")
        # data **= 2
        return data


class DataSource2D(DataSource):
    def __init__(self):
        super().__init__(shape=(256, 256, 1), ax_lims=((-1600, 1600), (-1600, 1600), (3200, 3200)))

    def post_process(self, data):
        # data -= data.min()
        # data += 0.01
        data.log_()
        return data


if __name__ == "__main__":
    ds = DataSource()
    ds.plot_arrays()