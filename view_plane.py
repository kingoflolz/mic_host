import queue
import torch
from vispy.visuals.transforms import STTransform

from datasource import DataSource2D
from vispy import app, scene

torch.set_grad_enabled(False)

source = DataSource2D()
data_queue = source.start()

canvas = scene.SceneCanvas(keys='interactive', show=True, vsync=True, size=(1024, 1024))
canvas.measure_fps()
view = canvas.central_widget.add_view()
# view.camera = scene.PanZoomCamera(aspect=1)
# view.camera.zoom(1/10000, (250, 200))
image = scene.visuals.Image(data_queue.get(), parent=view.scene, clim=(15, 22))
s = STTransform(scale=(4, 4, 4, 4))
image.transform = s
# cbar_widget = scene.ColorBarWidget(label="", clim=(17, 25),
#                                    cmap="viridis", orientation="top",
#                                    border_width=1)
# grid.add_widget(cbar_widget)

# grid.bgcolor = "#ffffff"


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
