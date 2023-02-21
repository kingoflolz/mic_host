import numpy as np
import torch

import matplotlib

from cal_model import CalModel

matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

# import mic_host
# import tqdm
# import numpy as np
# import queue
# import threading
#
# from utils import decimation, max_offset, clean_bad_mics
#
# chunks = []
#
# thread_run = True
#
# def read_thread(queue_out):
#     global thread_run
#     raw = mic_host.RawListener("192.168.1.1:5678", decimation)
#
#     while thread_run:
#         queue_out.put(raw.chunk(512 * 1024))
#
#
# def correlate_thread(queue_in, queue_out):
#     global thread_run
#     corr = mic_host.Correlator(32 * 1024)
#
#     while thread_run:
#         c = queue_in.get()
#         o = corr.correlate(c, max_offset)
#         queue_out.put(o)
#
#
# read_queue = queue.Queue()
# correlate_queue = queue.Queue()
#
# read_thread = threading.Thread(target=read_thread, args=(read_queue,), daemon=True).start()
# corr_thread = threading.Thread(target=correlate_thread, args=(read_queue, correlate_queue), daemon=True).start()
#
# offsets = []
# corrs = []
#
# for i in tqdm.tqdm(range(200), smoothing=1):
#     o, c = correlate_queue.get()
#
#     clean_bad_mics(o)
#     clean_bad_mics(c)
#
#     offsets.append(o)
#     corrs.append(c)
#
# thread_run = False
#
# offsets = np.stack(offsets)
# corrs = np.stack(corrs)
#
# np.save("calibration.npy", (offsets, corrs))
offsets, corrs = np.load("calibration.npy", allow_pickle=True)


cal = CalModel(len(offsets)).cuda()
opt = torch.optim.Adam(cal.parameters(), lr=10)
offsets = torch.Tensor(offsets).cuda()
corrs = torch.Tensor(corrs).cuda()

for i in range(200):
    opt.zero_grad()
    loss, time_loss, mic_pos_loss = cal(offsets, corrs)
    loss.backward()
    opt.step()
    print(i, loss.item(), time_loss.item(), mic_pos_loss.item())

torch.save(cal.state_dict(), "calibration.pt")

loss, time_loss, mic_pos_loss = cal(offsets, corrs)

mic_pos = cal.mic_pos.detach().cpu().numpy()
mic_pos_init = cal.starting_mic_pos.detach().cpu().numpy()
source_pos = cal.source_pos.detach().cpu().numpy()

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(mic_pos[:, 0], mic_pos[:, 1], mic_pos[:, 2], c="r")
ax.scatter(mic_pos_init[:, 0], mic_pos_init[:, 1], mic_pos_init[:, 2], c="b")
ax.plot(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], c="g")
ax.legend(["optimized", "initial", "sources"])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


for idx, (x, y, z) in enumerate(mic_pos):
    ax.text(x, y, z, f"{idx}")

plt.show()
print()