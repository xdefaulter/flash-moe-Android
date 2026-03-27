import json
import os
import subprocess
import tempfile
import unittest


class ValidatePackedLayoutCLITest(unittest.TestCase):
    def test_layout_and_files_validate(self):
        with tempfile.TemporaryDirectory(prefix="layout_test_") as td:
            layout = {
                "format": "flash_moe.packed_experts",
                "version": 1,
                "expert_size": 16,
                "num_layers": 2,
                "num_experts": 3,
                "layer_size": 48,
                "components": [{"name": "x", "offset": 0, "size": 16, "dtype": "U32"}],
                "layer_files": ["layer_00.bin", "layer_01.bin"],
            }
            layout_path = os.path.join(td, "layout.json")
            with open(layout_path, "w") as f:
                json.dump(layout, f)

            for name in layout["layer_files"]:
                with open(os.path.join(td, name), "wb") as f:
                    f.write(b"0" * layout["layer_size"])

            out = subprocess.check_output(
                ["python", "validate_packed_experts_layout.py", layout_path, "--check-files"],
                text=True,
            )
            self.assertIn("is valid", out)
            self.assertIn("all layer files are present", out)


if __name__ == "__main__":
    unittest.main()
