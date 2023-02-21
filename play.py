import mic_host
import tqdm
import numpy as np
import queue
import threading

from utils import decimation, max_offset

chunks = []


def read_thread(queue_out):
    raw = mic_host.RawListener("192.168.1.1:5678", decimation)

    while True:
        queue_out.put(raw.chunk(512 * 1024))


def correlate_thread(queue_in, queue_out):
    corr = mic_host.Correlator(32 * 1024)

    while True:
        c = queue_in.get()
        o = corr.correlate(c, max_offset)
        queue_out.put(o)


read_queue = queue.Queue()
correlate_queue = queue.Queue()

read_thread = threading.Thread(target=read_thread, args=(read_queue,), daemon=True).start()
corr_thread = threading.Thread(target=correlate_thread, args=(read_queue, correlate_queue), daemon=True).start()

offsets = []
corrs = []

for i in tqdm.tqdm(range(400), smoothing=1):
    o, c = correlate_queue.get()

    # clean_bad_mics(o)
    # clean_bad_mics(c)

    offsets.append(o)
    corrs.append(c)

# np.save("offsets.npy", np.stack(offsets))
# np.save("corrs.npy", np.stack(corrs))
# exit()