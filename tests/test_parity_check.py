import os
import subprocess
import tempfile
import unittest


class ParityCheckCLITest(unittest.TestCase):
    def test_tokens_mode_outputs_ratio(self):
        with tempfile.TemporaryDirectory(prefix="parity_tokens_") as td:
            ref = os.path.join(td, "ref.jsonl")
            tst = os.path.join(td, "tst.jsonl")
            with open(ref, "w") as f:
                f.write('{"step":0,"token":10}\n{"step":1,"token":20}\n')
            with open(tst, "w") as f:
                f.write('{"step":0,"token":10}\n{"step":1,"token":21}\n')

            out = subprocess.check_output(
                ["python", "tools/parity_check.py", "--mode", "tokens", "--ref", ref, "--test", tst],
                text=True,
            )
            self.assertIn("token_exact_match: 1/2", out)

    def test_router_mode_outputs_exact_and_jaccard(self):
        with tempfile.TemporaryDirectory(prefix="parity_router_") as td:
            ref = os.path.join(td, "ref.jsonl")
            tst = os.path.join(td, "tst.jsonl")
            with open(ref, "w") as f:
                f.write('{"step":0,"layer":0,"topk":[1,2,3,4]}\n')
            with open(tst, "w") as f:
                f.write('{"step":0,"layer":0,"topk":[1,2,4,9]}\n')

            out = subprocess.check_output(
                ["python", "tools/parity_check.py", "--mode", "router", "--ref", ref, "--test", tst],
                text=True,
            )
            self.assertIn("router_exact_match: 0/1", out)
            self.assertIn("router_mean_jaccard: 0.6000", out)


if __name__ == "__main__":
    unittest.main()
