from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_setup_module():
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("wan22_setup", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {setup_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_setup_module()


class TorchLaneSupportedTests(unittest.TestCase):
    def test_accepts_compact_and_dotted_cu128_or_newer(self):
        supported = (
            "",
            "128",
            "12.8",
            "129",
            "12.9",
            "13",
            "13.0",
            "130",
            "13.3",
            " 128 ",
        )
        for value in supported:
            with self.subTest(value=value):
                self.assertTrue(setup.torch_lane_supported(value))

    def test_rejects_older_lanes(self):
        unsupported = ("124", "12.4", "121", "12.1", "118", "11.8")
        for value in unsupported:
            with self.subTest(value=value):
                self.assertFalse(setup.torch_lane_supported(value))

    def test_rejects_malformed_or_ambiguous_values(self):
        malformed = (
            "abc",
            "1.28",
            "12.8.0",
            "12.80",
            "1280",
            "cu128",
            "-128",
            "12,8",
            ".",
        )
        for value in malformed:
            with self.subTest(value=value):
                self.assertFalse(setup.torch_lane_supported(value))


if __name__ == "__main__":
    unittest.main()
