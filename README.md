# Phased Array Microphone Host Software

For more detail on the project see links below:

- [Blogpost](https://benwang.dev/2023/02/26/Phased-Array-Microphone.html)
- [FPGA gateware](https://github.com/kingoflolz/mic_gateware)
- [PCB layout and schematics, mechanical components](https://github.com/kingoflolz/mic_hardware)

## Code structure
- Rust module (packet capture, CIC filtering and calibration cross correlation routines): `src/`
- Calibration entrypoint: `calibrate.py`
- 3D volume visualization entrypoint: `view_volume.py`
- 2D visualization entrypoint: `view_plane.py`
- Audio recording entrypoint: `record_audio.py`
- Others:
  - `utils.py`: Misc utilities
  - `kernel.py`: Triton kernel and tests
  - `datasource.py`: Packet capture and bulk DAS abstraction for python
  - `cal_model.py`: Array calibration model
