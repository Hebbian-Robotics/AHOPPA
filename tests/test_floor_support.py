"""Storage-round scenes use the flat ground collider and leashed raw facing."""

import json
import unittest
from pathlib import Path

from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.shutdown_task import STORAGE_CONDITIONS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class FloorSupportTests(unittest.TestCase):
    def test_storage_round_uses_the_flat_ground_collider(self) -> None:
        for condition in STORAGE_CONDITIONS:
            with self.subTest(condition=condition):
                scene = GroundWorkcellScene.model_validate_json(
                    (CONFIGS / f"shutdown-storage-{condition}.json").read_text()
                )
                self.assertEqual(scene.floor_support, "flat-ground-collider")
                self.assertEqual(scene.floor_collider_path, "/World/Ground/geometry/mesh")
                self.assertFalse(scene.stepped_turn_tool)
                self.assertTrue(scene.leashed_facing)

    def test_scenes_without_floor_support_keep_the_room_floor(self) -> None:
        scene = json.loads((CONFIGS / "shutdown-storage-none-open.json").read_text())
        del scene["floor_support"]
        parsed = GroundWorkcellScene.model_validate(scene)
        self.assertEqual(parsed.floor_support, "room-floor-mesh")
        self.assertTrue(parsed.floor_collider_path.endswith("Towel_Room01_floor_bottom"))

    def test_unknown_floor_support_is_rejected(self) -> None:
        scene = json.loads((CONFIGS / "shutdown-storage-none-open.json").read_text())
        scene["floor_support"] = "room-and-ground"
        with self.assertRaises(ValueError):
            GroundWorkcellScene.model_validate(scene)


if __name__ == "__main__":
    unittest.main()
