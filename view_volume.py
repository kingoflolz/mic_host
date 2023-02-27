import queue
import numpy as np
import torch

from datasource import DataSource3D

torch.set_grad_enabled(False)

from vispy import app, scene
from vispy.color import BaseColormap
from vispy.visuals.transforms import STTransform


source = DataSource3D()
data_queue = source.start()

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', show=True, vsync=True)
canvas.measure_fps()
view = canvas.central_widget.add_view()


class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


point_cloud = scene.visuals.Markers(parent=view.scene)
# Create the volume visuals, only one is visible
volume1 = scene.visuals.Volume(data_queue.get(),
                               parent=view.scene,
                               method="additive",
                               interpolation="gaussian",
                               )
volume1.cmap = TransFire()
# volume1.cmap = TransGrays()

mic_pos = source.cal.mic_pos.cpu().numpy() / (1600 / 24)
mic_pos += np.array([24, 24, 0])
mic_pos = mic_pos[:, [2, 0, 1]]
point_cloud.set_data(mic_pos, edge_width=0, face_color=(0, 1, 0, 1), size=3)
point_cloud.order = -1

canvas._draw_order.clear()
canvas.update()

fov = 60.
view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                            name='Turntable')

# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(view.camera.roll, (0, 0, 1))
        axis.transform.rotate(view.camera.elevation, (1, 0, 0))
        axis.transform.rotate(view.camera.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.update()


def update(ev):
    global volume1, data_queue
    try:
        d = data_queue.get(block=False)
        volume1.set_data(d)
        volume1.update()
    except queue.Empty:
        pass


timer = app.Timer()
timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    app.run()
