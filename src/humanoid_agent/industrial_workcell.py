"""Build a reproducible industrial appearance from the fixed task's collision boxes."""

from __future__ import annotations

import hashlib
import importlib
import itertools
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from cyclopts import App, Parameter
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.logging_config import configure_logging

MATERIALS = {
    "powder_coat": ((0.28, 0.32, 0.33), 0.65, 0.25),
    "door": ((0.60, 0.64, 0.63), 0.55, 0.25),
    "graphite": ((0.055, 0.065, 0.07), 0.65, 0.1),
    "steel": ((0.46, 0.49, 0.5), 0.32, 0.8),
    "wall": ((0.64, 0.65, 0.63), 0.9, 0.0),
    "yellow": ((0.88, 0.61, 0.08), 0.8, 0.0),
}


def usd_modules() -> tuple[Any, Any, Any]:
    """Use the pinned renderer runtime's bindings, without another USD installation."""
    try:
        return (
            importlib.import_module("pxr.UsdGeom"),
            importlib.import_module("pxr.UsdShade"),
            importlib.import_module("pxr.Sdf"),
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Generate scene assets with the pinned Isaac Python runtime (OpenUSD required)"
        ) from error


def author_mesh(
    stage: Any, name: str, points: Any, counts: list[int], indices: list[int], material: str
) -> Any:
    geometry, shading, _ = usd_modules()
    mesh = geometry.Mesh.Define(stage, "/IndustrialWorkcell/" + name)
    # Preserve the frozen appearance's eight-significant-digit coordinate precision.
    mesh.CreatePointsAttr([[float(f"{value:.8g}") for value in point] for point in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    shading.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        shading.Material.Get(stage, "/IndustrialWorkcell/Materials/" + material)
    )
    return mesh


def surface_material(stage: Any, name: str, roughness: float) -> tuple[Any, Any]:
    _, shading, types = usd_modules()
    material_path = "/IndustrialWorkcell/Materials/" + name
    material = shading.Material.Define(stage, material_path)
    surface = shading.Shader.Define(stage, material_path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", types.ValueTypeNames.Float).Set(roughness)
    surface.CreateOutput("surface", types.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material, surface


def beveled_box(
    stage: Any, name: str, center: list[float], size: list[float], material: str
) -> Any:
    half = np.asarray(size) / 2
    bevel = min(0.012, float(np.min(size)) * 0.15)
    vertices = []
    for signs in itertools.product([-1, 1], repeat=3):
        for outer_axis in range(3):
            point = half - bevel
            point[outer_axis] = half[outer_axis]
            vertices.append(np.asarray(signs) * point + center)
    points = np.asarray(vertices)
    hull = ConvexHull(points)
    triangles: list[list[int]] = hull.simplices.tolist()
    for index, triangle in enumerate(triangles):
        normal = np.cross(
            points[triangle[1]] - points[triangle[0]], points[triangle[2]] - points[triangle[0]]
        )
        if np.dot(normal, hull.equations[index, :3]) < 0:
            triangles[index] = [triangle[0], triangle[2], triangle[1]]
    mesh = author_mesh(
        stage,
        name,
        points.tolist(),
        [3] * len(triangles),
        [index for triangle in triangles for index in triangle],
        material,
    )
    mesh.CreateNormalsAttr(
        [[float(f"{value:.8g}") for value in normal] for normal in hull.equations[:, :3]]
    )
    mesh.SetNormalsInterpolation("uniform")
    return mesh


def label_material(stage: Any, name: str, texture: str) -> None:
    _, shading, types = usd_modules()
    _, surface = surface_material(stage, name, 0.8)
    material_path = "/IndustrialWorkcell/Materials/" + name
    texture_shader = shading.Shader.Define(stage, material_path + "/Texture")
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", types.ValueTypeNames.Asset).Set(
        types.AssetPath("./" + texture)
    )
    texture_shader.CreateInput("sourceColorSpace", types.ValueTypeNames.Token).Set("sRGB")
    texture_shader.CreateOutput("rgb", types.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", types.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb"
    )
    coordinates = shading.Shader.Define(stage, material_path + "/Coordinates")
    coordinates.CreateIdAttr("UsdPrimvarReader_float2")
    coordinates.CreateInput("varname", types.ValueTypeNames.Token).Set("st")
    coordinates.CreateOutput("result", types.ValueTypeNames.Float2)
    texture_shader.CreateInput("st", types.ValueTypeNames.Float2).ConnectToSource(
        coordinates.ConnectableAPI(), "result"
    )


def printed_label(stage: Any, name: str, center: list[float], width: float, height: float) -> Any:
    horizontal, lateral, vertical = center
    points = [
        [horizontal, lateral + width / 2, vertical - height / 2],
        [horizontal, lateral - width / 2, vertical - height / 2],
        [horizontal, lateral - width / 2, vertical + height / 2],
        [horizontal, lateral + width / 2, vertical + height / 2],
    ]
    mesh = author_mesh(stage, name, points, [4], [0, 1, 2, 3], name)
    geometry, _, types = usd_modules()
    geometry.PrimvarsAPI(mesh).CreatePrimvar(
        "st", types.ValueTypeNames.TexCoord2fArray, "vertex"
    ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    return mesh


def build_industrial_assets(scene_path: Path, output: Path) -> None:
    definition = GroundWorkcellScene.model_validate_json(scene_path.read_text())
    geometry, _, types = usd_modules()
    output.mkdir(parents=True, exist_ok=False)
    usd: Any = importlib.import_module("pxr.Usd")
    stage = usd.Stage.CreateNew(str(output / "industrial-workcell.usda"))
    stage.SetDefaultPrim(geometry.Xform.Define(stage, "/IndustrialWorkcell").GetPrim())
    geometry.SetStageUpAxis(stage, "Z")
    geometry.SetStageMetersPerUnit(stage, 1)
    geometry.Scope.Define(stage, "/IndustrialWorkcell/Materials")
    for name, (color, roughness, metallic) in MATERIALS.items():
        _, surface = surface_material(stage, name, roughness)
        surface.CreateInput("diffuseColor", types.ValueTypeNames.Color3f).Set(color)
        surface.CreateInput("metallic", types.ValueTypeNames.Float).Set(metallic)
    pieces = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
    for box in definition.boxes:
        center, size = list(box.center_m), list(box.full_size_m)
        material = "wall" if box.name == "back_wall" else "powder_coat"
        if box.name in {"panel_support", "panel_base"}:
            material = "steel"
        pieces.append(beveled_box(stage, box.name, center, size, material))
        front = center[0] - size[0] / 2
        if "cabinet" in box.name:
            pieces.append(
                beveled_box(
                    stage,
                    box.name + "_door",
                    [front - 0.003, center[1], center[2] + 0.02],
                    [0.008, size[1] - 0.045, size[2] - 0.16],
                    "door",
                )
            )
            pieces.append(
                beveled_box(
                    stage,
                    box.name + "_handle",
                    [front - 0.018, center[1] - 0.15, center[2] + 0.08],
                    [0.022, 0.018, 0.18],
                    "steel",
                )
            )
            pieces.append(
                beveled_box(
                    stage,
                    box.name + "_plinth",
                    [front - 0.002, center[1], 0.045],
                    [0.006, size[1] - 0.01, 0.08],
                    "graphite",
                )
            )
            for vent in range(4):
                pieces.append(
                    beveled_box(
                        stage,
                        box.name + f"_vent_{vent}",
                        [front - 0.009, center[1], center[2] + 0.38 + vent * 0.018],
                        [0.002, 0.28, 0.006],
                        "graphite",
                    )
                )
            label = box.name + "_label"
            title = "TOOLS 01" if box.name.startswith("left") else "SUPPLIES 02"
            label_center, label_size = [front - 0.009, center[1], center[2] + 0.25], (0.31, 0.065)
        elif box.name == "panel":
            pieces.append(
                beveled_box(
                    stage,
                    "panel_bezel",
                    [front - 0.002, center[1], center[2]],
                    [0.008, size[1] - 0.025, size[2] - 0.025],
                    "graphite",
                )
            )
            label, title = "panel_label", "CONTROL 01"
            label_center, label_size = [front - 0.007, center[1], center[2] + 0.17], (0.35, 0.065)
        else:
            continue
        image = Image.new("RGB", (640, 128), (228, 231, 225))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 15, 128), fill=(226, 172, 35))
        draw.text((320, 64), title, fill=(29, 34, 35), font=font, anchor="mm")
        filename = label + ".png"
        image.save(output / filename)
        label_material(stage, label, filename)
        pieces.append(printed_label(stage, label, label_center, *label_size))
    for index, lateral in enumerate([-0.93, 0.93]):
        pieces.append(
            beveled_box(
                stage, f"aisle_line_{index}", [1.35, lateral, 0.002], [3.8, 0.035, 0.002], "yellow"
            )
        )
    wall = next(box for box in definition.boxes if box.name == "back_wall")
    for index, lateral in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0]):
        pieces.append(
            beveled_box(
                stage,
                f"wall_seam_{index}",
                [wall.center_m[0] - wall.full_size_m[0] / 2 - 0.001, lateral, 1.0],
                [0.002, 0.008, 1.94],
                "graphite",
            )
        )
    stage.GetRootLayer().Save()
    atomic_json(
        output / "manifest.json",
        {
            "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "mesh_count": len(pieces),
            "visual_meshes_have_collisions": False,
            "collision_owner": "Task boxes and marker instantiated by the existing workcell factory",
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir())
            },
        },
    )


def main(
    *,
    cli_scene: Annotated[Path, Parameter(name="--scene")],
    cli_output: Annotated[Path, Parameter(name="--output")],
) -> None:
    build_industrial_assets(cli_scene, cli_output)


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
