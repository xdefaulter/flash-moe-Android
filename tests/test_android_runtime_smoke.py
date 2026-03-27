import json
import os
import subprocess
import tempfile
import unittest


class AndroidRuntimeSmokeTest(unittest.TestCase):
    def test_init_generate_stats_shutdown(self):
        with tempfile.TemporaryDirectory(prefix="android_rt_") as td:
            model_dir = td
            packed = os.path.join(model_dir, "packed_experts")
            os.makedirs(packed, exist_ok=True)

            with open(os.path.join(model_dir, "model_config.json"), "w") as f:
                json.dump(
                    {
                        "num_layers": 2,
                        "experts_per_token": 2,
                        "vocab_size": 128,
                        "hidden_dim": 32,
                        "moe_intermediate": 32,
                        "num_attention_heads": 4,
                        "head_dim": 8,
                        "context_len": 32,
                    },
                    f,
                )

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
            with open(os.path.join(packed, "layout.json"), "w") as f:
                json.dump(layout, f)

            for i, name in enumerate(layout["layer_files"]):
                with open(os.path.join(packed, name), "wb") as f:
                    f.write(bytes([(i + 1) * 7 + (j % 200) for j in range(layout["layer_size"])]))
            with open(os.path.join(model_dir, "model_weights.bin"), "wb") as f:
                f.write(bytes((j % 251) for j in range(128 * 1024)))

            main_cpp = os.path.join(td, "smoke.cpp")
            with open(main_cpp, "w") as f:
                f.write(
                    '#include <cstdio>\n'
                    '#include "android/infer_android.h"\n'
                    'int main(int argc, char** argv) {\n'
                    '  int rc = fm_android_init(argv[1], nullptr);\n'
                    '  if (rc != 0) return rc;\n'
                    '  int in[2] = {11, 22};\n'
                    '  int out[4] = {0};\n'
                    '  int n = fm_android_generate(in, 2, 4, out, 4);\n'
                    '  if (n != 4) return 7;\n'
                    '  fm_android_stats_t s{};\n'
                    '  if (fm_android_get_stats(&s) != 0) return 8;\n'
                    '  if (s.expert_reads == 0 || s.expert_bytes == 0) return 9;\n'
                    '  fm_android_shutdown();\n'
                    '  return 0;\n'
                    '}\n'
                )

            bin_path = os.path.join(td, "rt_smoke")
            subprocess.check_call(
                [
                    "g++",
                    "-std=c++17",
                    "-I.",
                    main_cpp,
                    "android/infer_android.cpp",
                    "android/vulkan_backend.cpp",
                    "-o",
                    bin_path,
                ]
            )
            subprocess.check_call([bin_path, model_dir])


if __name__ == "__main__":
    unittest.main()
