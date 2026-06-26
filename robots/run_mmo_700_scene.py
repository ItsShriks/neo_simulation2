#!/usr/bin/env python3
from pathlib import Path
import tempfile

import mujoco
import trimesh
from mujoco import viewer


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "mmo_700.xml"


def get_scaled_body_mesh() -> Path:
    source_mesh = SCRIPT_DIR / "mmo_700" / "meshes" / "MPO-700-BODY.stl"
    scaled_mesh = SCRIPT_DIR / "mmo_700" / "meshes" / "MPO-700-BODY_scaled.stl"
    if not scaled_mesh.exists():
        mesh = trimesh.load_mesh(source_mesh)
        mesh.apply_scale(0.001)
        mesh.export(scaled_mesh)
    return scaled_mesh


def build_scene_xml() -> str:
    mesh_path = get_scaled_body_mesh()
    xml_text = f"""
    <mujoco model="mmo_700_scene">
      <compiler angle="radian" />
      <option gravity="0 0 -9.81" />
      <asset>
        <mesh name="body_mesh" file="{mesh_path}" />
      </asset>
      <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true" />
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.6 0.6 0.6 1" />

        <body name="robot_base" pos="0 0 0.15">
          <freejoint name="robot_free" />
          <geom type="mesh" mesh="body_mesh" rgba="0.8 0.8 0.8 1" />
          <body name="cabinet" pos="0.08 0 0.35">
            <geom type="box" size="0.08 0.12 0.12" rgba="0.25 0.25 0.30 1" />
          </body>
          <body name="arm_mount" pos="0.10 0 0.45">
            <geom type="cylinder" size="0.035 0.06" rgba="0.55 0.55 0.55 1" />
            <body name="upper_arm" pos="0 0 0.09">
              <geom type="box" size="0.03 0.03 0.16" rgba="0.45 0.45 0.45 1" />
            </body>
            <body name="forearm" pos="0 0 0.24">
              <geom type="box" size="0.025 0.025 0.12" rgba="0.55 0.55 0.55 1" />
            </body>
          </body>
          <body name="wheel_front_left" pos="0.24 0.18 0.06">
            <geom type="cylinder" size="0.05 0.04" rgba="0.12 0.12 0.12 1" />
          </body>
          <body name="wheel_front_right" pos="0.24 -0.18 0.06">
            <geom type="cylinder" size="0.05 0.04" rgba="0.12 0.12 0.12 1" />
          </body>
          <body name="wheel_back_left" pos="-0.24 0.18 0.06">
            <geom type="cylinder" size="0.05 0.04" rgba="0.12 0.12 0.12 1" />
          </body>
          <body name="wheel_back_right" pos="-0.24 -0.18 0.06">
            <geom type="cylinder" size="0.05 0.04" rgba="0.12 0.12 0.12 1" />
          </body>
        </body>

        <body name="table" pos="0.8 0.0 0.4">
          <geom type="box" size="0.4 0.6 0.02" rgba="0.5 0.3 0.1 1" />
          <geom type="box" size="0.03 0.03 0.2" pos="0.3 0.4 0.2" rgba="0.35 0.2 0.1 1" />
          <geom type="box" size="0.03 0.03 0.2" pos="-0.3 0.4 0.2" rgba="0.35 0.2 0.1 1" />
          <geom type="box" size="0.03 0.03 0.2" pos="0.3 -0.4 0.2" rgba="0.35 0.2 0.1 1" />
          <geom type="box" size="0.03 0.03 0.2" pos="-0.3 -0.4 0.2" rgba="0.35 0.2 0.1 1" />
        </body>

        <body name="static_box" pos="0.7 0.2 0.82">
          <geom type="box" size="0.08 0.08 0.08" rgba="1 0 0 1" />
        </body>
      </worldbody>
    </mujoco>
    """

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="mmo_700_scene_", suffix=".xml")
    Path(tmp_path).write_text(xml_text)
    return tmp_path


def main() -> None:
    scene_xml_path = build_scene_xml()
    print(f"Loaded scene from {MODEL_PATH}")
    print(f"Using temporary scene XML: {scene_xml_path}")
    print("Launching MuJoCo viewer...")

    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
