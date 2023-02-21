import queue
import torch

from datasource import DataSource2D
from vispy import app, scene

torch.set_grad_enabled(False)

source = DataSource2D()
data_queue = source.start()

canvas = scene.SceneCanvas(keys='interactive', show=True, vsync=True)
canvas.measure_fps()
grid = canvas.central_widget.add_grid()
view = grid.add_view()

image = scene.visuals.Image(data_queue.get(), parent=view.scene, clim=(15, 22))

# cbar_widget = scene.ColorBarWidget(label="", clim=(17, 25),
#                                    cmap="viridis", orientation="top",
#                                    border_width=1)
# grid.add_widget(cbar_widget)

grid.bgcolor = "#ffffff"


def update(ev):
    global volume1, data_queue
    try:
        d = data_queue.get(block=False)
        image.set_data(d)
        image.update()
    except queue.Empty:
        pass


timer = app.Timer()
timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    app.run()
