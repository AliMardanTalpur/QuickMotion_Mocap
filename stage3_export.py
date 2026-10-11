"""Stage 3: headless Blender IK rig, bake, and BVH/FBX export.

Run: blender -b --factory-startup --python stage3_export.py -- \
    --input 3d_points.npy --bvh motion.bvh --fbx motion.fbx

Accepts Human3.6M-17 or GVHMR's 24 SMPL-X camera joints. The 24-joint route
retains measured feet, collars, and the fuller spine. Small terminal hand
bones are synthesized. IK uses one bone per chain, so each measured segment
aims at its tracked joint while retaining a fixed rest-pose length.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from motion_processing import gaussian_time, reduce_footskate


@dataclass(frozen=True)
class Segment:
    bone: str
    start: int
    end: int
    parent: str


# YOLO/COCO input joint indices are converted by Stage 2's COCO-trained
# VideoPose3D checkpoint to Human3.6M order. The indices below address that
# *output*, not the original COCO order. For example, COCO left shoulder is 5,
# while Human3.6M left shoulder is 11 and drives Blender's LeftShoulder bone.
# Mid-spine/neck points (7..10) are inferred by VideoPose3D because COCO does
# not directly measure them. Blender names on the left are our exported rig.
SEGMENTS = [
    Segment("Spine", 0, 7, "Root"),
    Segment("Chest", 7, 8, "Spine"),
    Segment("Neck", 8, 9, "Chest"),
    Segment("Head", 9, 10, "Neck"),
    Segment("LeftHip", 0, 4, "Root"),
    Segment("LeftUpLeg", 4, 5, "LeftHip"),
    Segment("LeftLeg", 5, 6, "LeftUpLeg"),
    Segment("RightHip", 0, 1, "Root"),
    Segment("RightUpLeg", 1, 2, "RightHip"),
    Segment("RightLeg", 2, 3, "RightUpLeg"),
    Segment("LeftShoulder", 8, 11, "Chest"),
    Segment("LeftArm", 11, 12, "LeftShoulder"),
    Segment("LeftForeArm", 12, 13, "LeftArm"),
    Segment("RightShoulder", 8, 14, "Chest"),
    Segment("RightArm", 14, 15, "RightShoulder"),
    Segment("RightForeArm", 15, 16, "RightArm"),
]

# GVHMR provides the native SMPL-X body joints, so its export can retain more
# measured anatomy than the COCO/H36M-17 bridge. Indices 22 and 23 are facial
# joints in this model and are intentionally excluded from the body armature.
# These names are Blender bone names; the integers are SMPL-X joint indices.
SMPLX24_SEGMENTS = [
    Segment("Spine1", 0, 3, "Root"),
    Segment("Spine2", 3, 6, "Spine1"),
    Segment("Spine3", 6, 9, "Spine2"),
    Segment("Neck", 9, 12, "Spine3"),
    Segment("Head", 12, 15, "Neck"),
    Segment("LeftHip", 0, 1, "Root"),
    Segment("LeftKnee", 1, 4, "LeftHip"),
    Segment("LeftAnkle", 4, 7, "LeftKnee"),
    Segment("LeftFoot", 7, 10, "LeftAnkle"),
    Segment("RightHip", 0, 2, "Root"),
    Segment("RightKnee", 2, 5, "RightHip"),
    Segment("RightAnkle", 5, 8, "RightKnee"),
    Segment("RightFoot", 8, 11, "RightAnkle"),
    Segment("LeftCollar", 9, 13, "Spine3"),
    Segment("LeftShoulder", 13, 16, "LeftCollar"),
    Segment("LeftElbow", 16, 18, "LeftShoulder"),
    Segment("LeftWrist", 18, 20, "LeftElbow"),
    Segment("RightCollar", 9, 14, "Spine3"),
    Segment("RightShoulder", 14, 17, "RightCollar"),
    Segment("RightElbow", 17, 19, "RightShoulder"),
    Segment("RightWrist", 19, 21, "RightElbow"),
]


def blender_coordinates(points: np.ndarray) -> np.ndarray:
    # Camera space: x right, y down, z forward. Blender: X right, Y forward,
    # Z up. This axis permutation preserves handedness.
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)


def median_lengths(points: np.ndarray, segments: list[Segment] = SEGMENTS) -> dict[str, float]:
    result = {}
    for segment in segments:
        distance = np.linalg.norm(
            points[:, segment.end] - points[:, segment.start], axis=1
        )
        result[segment.bone] = max(float(np.median(distance)), 0.035)
    return result


def canonical_rest(length: dict[str, float], joint_count: int = 17) -> dict[int, Vector]:
    """Build a stable T-pose with the measured median segment lengths."""
    if joint_count == 24:
        p: dict[int, Vector] = {0: Vector((0, 0, 0))}
        for start, end, bone in ((0, 3, "Spine1"), (3, 6, "Spine2"),
                                 (6, 9, "Spine3"), (9, 12, "Neck"),
                                 (12, 15, "Head")):
            p[end] = p[start] + Vector((0, 0, length[bone]))
        for side, sign, hip, knee, ankle, foot in (
            ("Left", 1, 1, 4, 7, 10), ("Right", -1, 2, 5, 8, 11)
        ):
            p[hip] = p[0] + Vector((sign * length[f"{side}Hip"], 0, 0))
            p[knee] = p[hip] + Vector((0, 0, -length[f"{side}Knee"]))
            p[ankle] = p[knee] + Vector((0, 0, -length[f"{side}Ankle"]))
            p[foot] = p[ankle] + Vector((0, -length[f"{side}Foot"], 0))
        for side, sign, collar, shoulder, elbow, wrist in (
            ("Left", 1, 13, 16, 18, 20), ("Right", -1, 14, 17, 19, 21)
        ):
            p[collar] = p[9] + Vector((sign * length[f"{side}Collar"], 0, 0))
            p[shoulder] = p[collar] + Vector((sign * length[f"{side}Shoulder"], 0, 0))
            p[elbow] = p[shoulder] + Vector((sign * length[f"{side}Elbow"], 0, 0))
            p[wrist] = p[elbow] + Vector((sign * length[f"{side}Wrist"], 0, 0))
        return p
    if joint_count != 17:
        raise ValueError(f"Unsupported skeleton joint count: {joint_count}")
    p: dict[int, Vector] = {0: Vector((0, 0, 0))}
    for a, b, name in ((0, 7, "Spine"), (7, 8, "Chest"),
                       (8, 9, "Neck"), (9, 10, "Head")):
        p[b] = p[a] + Vector((0, 0, length[name]))
    # A front-facing subject looks toward Blender -Y: anatomical left is +X.
    for sign, hip, knee, ankle, prefix in (
        (1, 4, 5, 6, "Left"), (-1, 1, 2, 3, "Right")
    ):
        p[hip] = p[0] + Vector((sign * length[f"{prefix}Hip"], 0, 0))
        p[knee] = p[hip] + Vector((0, 0, -length[f"{prefix}UpLeg"]))
        p[ankle] = p[knee] + Vector((0, 0, -length[f"{prefix}Leg"]))
    for sign, shoulder, elbow, wrist, prefix in (
        (1, 11, 12, 13, "Left"), (-1, 14, 15, 16, "Right")
    ):
        p[shoulder] = p[8] + Vector((sign * length[f"{prefix}Shoulder"], 0, 0))
        p[elbow] = p[shoulder] + Vector((sign * length[f"{prefix}Arm"], 0, 0))
        p[wrist] = p[elbow] + Vector((sign * length[f"{prefix}ForeArm"], 0, 0))
    return p


def make_armature(rest: dict[int, Vector], scale: float,
                  segments: list[Segment] = SEGMENTS) -> bpy.types.Object:
    armature = bpy.data.armatures.new("MocapSkeletonData")
    rig = bpy.data.objects.new("MocapSkeleton", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature.edit_bones.new("Root")
    # Root's tail is the pelvis pivot. Every segment can then connect to its
    # parent, so the exported BVH only needs translation channels on Root.
    root.tail = rest[0]
    root.head = rest[0] - Vector((0, 0, max(0.05 * scale, 0.03)))
    for segment in segments:
        bone = armature.edit_bones.new(segment.bone)
        bone.head = rest[segment.start]
        bone.tail = rest[segment.end]
        bone.parent = armature.edit_bones[segment.parent]
        bone.use_connect = True
    # FBX importers can choose an arbitrary length for a terminal Head bone.
    # A connected child preserves the measured neck-to-head segment on import.
    head_tip = armature.edit_bones.new("HeadTip")
    head_index = 15 if len(rest) == 22 else 10
    head_tip.head = rest[head_index]
    head_tip.tail = rest[head_index] + Vector((0, 0, 0.10 * scale))
    head_tip.parent = armature.edit_bones["Head"]
    head_tip.use_connect = True
    # COCO/H36M has no foot tip or finger landmarks. Terminal bones are visual
    # placeholders and keep conventional humanoid end effectors in the export.
    if len(rest) == 22:
        terminal_joints = (("Left", 20, "LeftWrist"),
                           ("Right", 21, "RightWrist"))
        # FBX importers may assign a new length to a leaf bone. A connected
        # child fixes each measured ankle-to-foot-tip length on reimport.
        for side, foot_index in (("Left", 10), ("Right", 11)):
            foot_tip = armature.edit_bones.new(f"{side}FootTip")
            foot_tip.head = rest[foot_index]
            foot_tip.tail = foot_tip.head + Vector((0, -0.08 * scale, 0))
            foot_tip.parent = armature.edit_bones[f"{side}Foot"]
            foot_tip.use_connect = True
    else:
        terminal_joints = (("Left", 13, "LeftForeArm"),
                           ("Right", 16, "RightForeArm"))
        for side, ankle in (("Left", 6), ("Right", 3)):
            foot = armature.edit_bones.new(f"{side}Foot")
            foot.head = rest[ankle]
            foot.tail = foot.head + Vector((0, -0.15 * scale, -0.025 * scale))
            foot.parent = armature.edit_bones[f"{side}Leg"]
            foot.use_connect = True
    for side, wrist, parent in terminal_joints:
        hand = armature.edit_bones.new(f"{side}Hand")
        hand.head = rest[wrist]
        hand.tail = hand.head + Vector(((1 if side == "Left" else -1) * 0.12 * scale, 0, 0))
        hand.parent = armature.edit_bones[parent]
        hand.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.display_type = "STICK"
    rig.show_in_front = True
    return rig


def create_targets(rig: bpy.types.Object, points: np.ndarray,
                   segments: list[Segment] = SEGMENTS) -> list[bpy.types.Object]:
    targets = []
    for joint in range(points.shape[1]):
        empty = bpy.data.objects.new(f"Target_{joint:02d}", None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_size = 0.04
        targets.append(empty)
    for segment in segments:
        # The IK target is the 3D coordinate corresponding to this bone's tail.
        # With chain_count=1, each bone follows its mapped joint target while
        # its length remains the fixed length built from the 3D sequence.
        constraint = rig.pose.bones[segment.bone].constraints.new("IK")
        constraint.name = "Track3DJoint"
        constraint.target = targets[segment.end]
        constraint.chain_count = 1
        constraint.use_stretch = False
    for frame_index, frame_points in enumerate(points, start=1):
        for joint, empty in enumerate(targets):
            empty.location = frame_points[joint].tolist()
            empty.keyframe_insert(data_path="location", frame=frame_index)
        root = rig.pose.bones["Root"]
        # PoseBone.location is in the bone's *local rest basis*, not Blender
        # world XYZ. A vertical root bone maps local Y onto world Z. Assigning
        # world XYZ directly made the original animation lie on its side.
        world_delta = Vector(frame_points[0]) - root.bone.tail_local
        root.location = root.bone.matrix_local.to_3x3().inverted() @ world_delta
        root.keyframe_insert(data_path="location", frame=frame_index)
    return targets


def bake_and_export(
    rig: bpy.types.Object, bvh: Path, fbx: Path,
    points: np.ndarray, fps: float, motion_report: dict | None = None,
    segments: list[Segment] = SEGMENTS,
) -> None:
    frames = len(points)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / fps
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.nla.bake(
        frame_start=1, frame_end=frames, step=1,
        only_selected=False, visual_keying=True, clear_constraints=True,
        use_current_action=False, bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(1)
    if rig.animation_data is None or rig.animation_data.action is None:
        raise RuntimeError("Blender did not create a baked armature action")
    # Confirm the baked action still follows the input. This would have caught
    # the previous local-vs-world root-location bug before exporting it.
    errors = []
    root_errors = []
    for frame_index in range(1, frames + 1):
        scene.frame_set(frame_index)
        root_errors.append(float(np.linalg.norm(
            np.asarray(rig.pose.bones["Root"].tail) - points[frame_index - 1, 0]
        )))
        for segment in segments:
            errors.append(float(np.linalg.norm(
                np.asarray(rig.pose.bones[segment.bone].tail)
                - points[frame_index - 1, segment.end]
            )))
    if max(root_errors) > 0.02:
        raise RuntimeError(f"Baked root is displaced from 3D targets by {max(root_errors):.3f} m")
    if float(np.median(errors)) > 0.20:
        raise RuntimeError("Baked skeleton is far from the 3D joint targets")
    report = {
        "frames": frames,
        "fps": fps,
        "root_max_error_m": max(root_errors),
        "joint_median_error_m": float(np.median(errors)),
        "joint_p95_error_m": float(np.percentile(errors, 95)),
        "joint_max_error_m": max(errors),
    }
    if motion_report:
        report.update(motion_report)
    bvh.parent.mkdir(parents=True, exist_ok=True)
    fbx.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=str(bvh.resolve()), frame_start=1, frame_end=frames,
        root_transform_only=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(fbx.resolve()), use_selection=True,
        object_types={"ARMATURE"}, add_leaf_bones=False,
        bake_anim=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False, bake_anim_simplify_factor=0.0,
    )
    if not bvh.is_file() or not fbx.is_file():
        raise RuntimeError("Blender export did not create both files")
    # Validate the deliverable, not only the in-memory rig: an FBX importer can
    # reinterpret leaf-bone lengths or axes even when the baked action is good.
    known_objects = set(bpy.data.objects.keys())
    # Blender's FBX importer defaults to anim_offset=1, shifting all imported
    # keys one frame later. Compare frame t with frame t by disabling it here.
    bpy.ops.import_scene.fbx(filepath=str(fbx.resolve()), anim_offset=0.0)
    imported = [obj for obj in bpy.data.objects if obj.name not in known_objects]
    imported_rigs = [obj for obj in imported if obj.type == "ARMATURE"]
    if len(imported_rigs) != 1:
        raise RuntimeError("FBX reimport did not produce exactly one armature")
    imported_rig = imported_rigs[0]
    reimport_errors = []
    reimport_by_bone = {segment.bone: [] for segment in segments}
    worst_frames = []
    for frame_index in range(1, frames + 1):
        scene.frame_set(frame_index)
        frame_errors = []
        for segment in segments:
            bone = imported_rig.pose.bones[segment.bone]
            position = imported_rig.matrix_world @ bone.tail
            error = float(np.linalg.norm(
                np.asarray(position) - points[frame_index - 1, segment.end]
            ))
            reimport_errors.append(error)
            reimport_by_bone[segment.bone].append(error)
            frame_errors.append(error)
        worst_frames.append(float(np.percentile(frame_errors, 95)))
    report["fbx_reimport_joint_median_error_m"] = float(np.median(reimport_errors))
    report["fbx_reimport_joint_p95_error_m"] = float(np.percentile(reimport_errors, 95))
    report["fbx_reimport_joint_max_error_m"] = max(reimport_errors)
    report["fbx_reimport_p95_error_by_bone_m"] = {
        bone: float(np.percentile(values, 95))
        for bone, values in reimport_by_bone.items()
    }
    report["fbx_reimport_worst_frames"] = [
        int(index + 1) for index in np.argsort(worst_frames)[-5:][::-1]
    ]
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    bvh.with_suffix(".quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["fbx_reimport_joint_p95_error_m"] > 0.08:
        raise RuntimeError(
            "Reimported FBX differs too much from the animated 3D targets; "
            f"see {bvh.with_suffix('.quality.json')} for per-bone errors"
        )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("3d_points.npy"))
    parser.add_argument("--bvh", type=Path, default=Path("motion.bvh"))
    parser.add_argument("--fbx", type=Path, default=Path("motion.fbx"))
    parser.add_argument("--fps", type=float, default=None, help="Override Stage 2 metadata FPS")
    parser.add_argument("--ground", choices=("auto", "stance", "hands-feet", "fixed"), default="auto",
                        help="Auto includes hands for image-guided acrobatics; fixed preserves jumps")
    parser.add_argument("--foot-lock", choices=("on", "off"), default="on",
                        help="Gently correct root XY during detected foot contacts")
    args = parser.parse_args(argv)
    points = np.load(args.input, allow_pickle=False)
    if points.ndim != 3 or points.shape[1:] not in ((17, 3), (24, 3)) or len(points) == 0:
        raise ValueError("Input must be (frames, 17, 3) H36M or (frames, 24, 3) SMPL-X")
    if not np.isfinite(points).all():
        raise ValueError("3D input contains non-finite coordinates")
    metadata_path = args.input.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    fps = args.fps or float(metadata.get("fps", 30))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    joint_count = points.shape[1]
    segments = SMPLX24_SEGMENTS if joint_count == 24 else SEGMENTS
    if joint_count == 24:
        # GVHMR camera joints contain absolute distance from the camera.
        # Rebase only the constant horizontal/depth origin; preserve all
        # frame-to-frame root travel and full SMPL-X body depth.
        points = points.copy()
        points[:, :, 0] -= float(np.median(points[:, 0, 0]))
        points[:, :, 2] -= float(np.median(points[:, 0, 2]))
    converted = blender_coordinates(points)
    # COCO wrists can be support points during handstands and floor work.
    # The Stage 2 image-guided mode is used for these out-of-domain poses, so
    # auto grounding includes wrists there; ordinary clips retain ankle stance.
    ground_mode = args.ground
    if ground_mode == "auto":
        ground_mode = "hands-feet" if metadata.get("pose_mode") == "image-guided" else "stance"
    if joint_count == 24:
        support_joints = (7, 8, 10, 11, 20, 21) if ground_mode == "hands-feet" else (7, 8, 10, 11)
    else:
        support_joints = (3, 6, 13, 16) if ground_mode == "hands-feet" else (3, 6)
    support_floor = np.min(converted[:, support_joints, 2], axis=1)
    if ground_mode != "fixed":
        floor = gaussian_time(support_floor.astype(np.float32), sigma=1.5)
        converted[:, :, 2] -= floor[:, None]
    else:
        converted[:, :, 2] -= float(np.percentile(support_floor, 5))
    motion_report = {"ground_mode": ground_mode}
    if args.foot_lock == "on":
        if joint_count == 24:
            raise ValueError("Foot lock currently accepts H36M-17 input; use --foot-lock off for SMPL-X")
        converted, motion_report = reduce_footskate(converted, fps)
        print("Foot contact:", motion_report)
    relative = converted - converted[:, :1]
    lengths = median_lengths(relative, segments)
    rest = canonical_rest(lengths, joint_count)
    leg_scale = (lengths["LeftKnee"] + lengths["LeftAnkle"] if joint_count == 24
                 else lengths["LeftUpLeg"] + lengths["LeftLeg"])
    rig = make_armature(rest, max(leg_scale, 0.5), segments)
    create_targets(rig, converted, segments)
    bake_and_export(rig, args.bvh, args.fbx, converted, fps, motion_report, segments)
    print(f"Exported {len(points)} frames at {fps:.3f} FPS: {args.bvh}, {args.fbx}")


if __name__ == "__main__":
    main()
