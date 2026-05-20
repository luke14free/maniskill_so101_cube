"""Render a single screenshot of the cube-in-bin scene from scene_config.yaml.

Iterate this way:
  1. Edit scene_config.yaml
  2. python render_scene.py
  3. open render.png and compare to your real photo
  4. Tweak YAML, repeat.
"""
import math
import os
import sys
from pathlib import Path

# macOS only: point SAPIEN at MoltenVK (brew install molten-vk vulkan-loader).
# On Linux/CUDA the system Vulkan loader is used — these env vars must NOT be set
# to a Mac path or SAPIEN will fail to find a rendering device.
if sys.platform == "darwin":
    os.environ.setdefault(
        "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
    )
    _existing_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if "/opt/homebrew/lib" not in _existing_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = (
            "/opt/homebrew/lib" + (":" + _existing_dyld if _existing_dyld else "")
        )

import numpy as np
import sapien
import yaml
from PIL import Image
from sapien import Pose
from transforms3d.euler import euler2quat

HERE = Path(__file__).parent
# Resolve the SO-100 URDF from the installed ManiSkill package, wherever pip put it.
import mani_skill as _ms

SO100_URDF = Path(_ms.__file__).parent / "assets/robots/so100/so100.urdf"
CFG = yaml.safe_load((HERE / "scene_config.yaml").read_text())


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Return a SAPIEN quaternion (w,x,y,z) for a camera at `eye` looking at `target`.
    Camera convention in SAPIEN: +X is the look direction."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=float)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, fwd)
    # rotation matrix with columns = [fwd, -right, new_up]  (SAPIEN camera +X fwd, -Y right, +Z up)
    R = np.stack([fwd, -right, new_up], axis=1)
    # convert rotation matrix to quaternion (w,x,y,z)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return [w, x, y, z]


def add_box(scene, name, half_size, pose, color):
    """Add a static box (visual+collision) to the scene."""
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size, material=color)
    actor = builder.build_static(name=name)
    actor.set_pose(pose)
    return actor


def add_cylinder(scene, name, radius, half_length, pose, color):
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(radius=radius, half_length=half_length)
    builder.add_cylinder_visual(radius=radius, half_length=half_length, material=color)
    actor = builder.build_static(name=name)
    actor.set_pose(pose)
    return actor


def add_hollow_cylinder(
    scene,
    name,
    radius,
    height,
    wall_thickness,
    color,
    base_xy,
    base_z,
    n_segments=32,
    inner_color=None,
):
    """Open-top cylindrical container approximated by N wall segments + a thin bottom disc.
    base_xy = (x, y) of the cylinder axis; base_z = z of the bin's OUTSIDE bottom.
    """
    builder = scene.create_actor_builder()
    half_h = height / 2
    # box length along the tangent (with slight overlap so segments don't leak gaps)
    seg_half_len = radius * math.sin(math.pi / n_segments) * 1.05
    for i in range(n_segments):
        a = (i + 0.5) * (2 * math.pi / n_segments)
        cx = radius * math.cos(a)
        cy = radius * math.sin(a)
        local_pose = Pose(
            [cx, cy, 0],
            euler2quat(0, 0, a + math.pi / 2),
        )
        builder.add_box_visual(
            pose=local_pose,
            half_size=[seg_half_len, wall_thickness / 2, half_h],
            material=color,
        )
        builder.add_box_collision(
            pose=local_pose,
            half_size=[seg_half_len, wall_thickness / 2, half_h],
        )
    # thin disc bottom — cylinder primitive rotated upright
    bottom_thick = 0.003
    upright = euler2quat(0, math.pi / 2, 0)
    bc = inner_color if inner_color is not None else color
    builder.add_cylinder_visual(
        pose=Pose([0, 0, -half_h + bottom_thick / 2], upright),
        radius=radius * 0.98,
        half_length=bottom_thick / 2,
        material=bc,
    )
    builder.add_cylinder_collision(
        pose=Pose([0, 0, -half_h + bottom_thick / 2], upright),
        radius=radius * 0.98,
        half_length=bottom_thick / 2,
    )
    actor = builder.build_static(name=name)
    actor.set_pose(Pose([base_xy[0], base_xy[1], base_z + half_h]))
    return actor


def build_scene():
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0.0, -0.3, -1.0], [0.9, 0.9, 0.9])
    scene.add_directional_light([1.0, 1.0, -0.5], [0.4, 0.4, 0.4])

    # ---- table top ----
    t = CFG["table"]
    tx, ty, tz = t["size"]
    table_pose = Pose([0, 0, t["top_z"] - tz / 2])
    add_box(scene, "table", [tx / 2, ty / 2, tz / 2], table_pose, t["color"])

    # ---- mat ----
    m = CFG["mat"]
    mx, my, mz = m["size"]
    mat_pose = Pose([m["center"][0], m["center"][1], t["top_z"] + mz / 2])
    add_box(scene, "mat", [mx / 2, my / 2, mz / 2], mat_pose, m["color"])
    mat_top_z = t["top_z"] + mz

    # ---- bin (hollow open-top container) ----
    b = CFG["bin"]
    add_hollow_cylinder(
        scene,
        "bin",
        radius=b["diameter"] / 2,
        height=b["height"],
        wall_thickness=b.get("wall_thickness", 0.003),
        color=b["color"],
        base_xy=b["position"],
        base_z=t["top_z"],
        n_segments=b.get("n_segments", 48),
        inner_color=b.get("inner_color", b["color"]),
    )

    # ---- cube (optional) ----
    if CFG.get("show_cube", True):
        c = CFG["cube"]
        s = c["size"]
        cube_pose = Pose([c["position"][0], c["position"][1], mat_top_z + s / 2])
        add_box(scene, "cube", [s / 2, s / 2, s / 2], cube_pose, c["color"])

    # ---- backdrop (red room of walls) ----
    bd = CFG["backdrop"]
    color = bd["color"]
    if "walls" in bd:
        for i, wall in enumerate(bd["walls"]):
            sx, sy, sz = wall["size"]
            add_box(
                scene,
                f"backdrop_{i}",
                [sx / 2, sy / 2, sz / 2],
                Pose(wall["center"]),
                color,
            )
    else:  # back-compat single-slab form
        bdx, bdy, bdz = bd["size"]
        add_box(
            scene, "backdrop", [bdx / 2, bdy / 2, bdz / 2], Pose(bd["position"]), color
        )

    # ---- robot (SO-100) ----
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(SO100_URDF))
    r = CFG["robot"]
    base_q = euler2quat(0, 0, math.radians(r["base_yaw_deg"]))
    robot.set_root_pose(Pose(r["base_pose"], base_q))
    # set rest pose if joint count matches
    qpos = np.zeros(robot.dof)
    rest = r["rest_qpos"]
    qpos[: min(len(rest), robot.dof)] = rest[: min(len(rest), robot.dof)]
    robot.set_qpos(qpos)

    # ---- camera ----
    cam_cfg = CFG["camera_scene"]
    cam_ent = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(
        width=cam_cfg["resolution"][0], height=cam_cfg["resolution"][1]
    )
    cam.set_fovy(math.radians(cam_cfg["fov_deg"]))
    cam_ent.add_component(cam)
    cam_q = look_at_quat(cam_cfg["position"], cam_cfg["target"])
    cam_ent.set_pose(Pose(cam_cfg["position"], cam_q))
    scene.add_entity(cam_ent)

    return scene, cam


def main():
    scene, cam = build_scene()
    # let physics settle one step so URDF visuals are placed
    scene.step()
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")  # H, W, 4 float32 in [0,1]
    rgb = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    out = HERE / "render.png"
    Image.fromarray(rgb).save(out)
    print(f"Saved: {out}  ({rgb.shape[1]}x{rgb.shape[0]})")


if __name__ == "__main__":
    main()
