import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ui_server


class ManifestTests(unittest.TestCase):
    def test_windows_lock_retries_and_keeps_latest_manifest(self):
        ui_server.RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ui_server.RUN_ROOT) as directory, patch.object(
            ui_server, "RUN_ROOT", Path(directory)
        ):
            job = {"id": "a" * 32, "status": "running"}
            real_replace = ui_server.os.replace
            calls = 0

            def briefly_locked(src, dst):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError(5, "Access is denied")
                return real_replace(src, dst)

            with patch.object(ui_server.os, "replace", side_effect=briefly_locked):
                ui_server.write_manifest(job)

            self.assertEqual(calls, 3)
            manifest = Path(directory) / job["id"] / "manifest.json"
            self.assertIn('"status": "running"', manifest.read_text())
            self.assertEqual(list(manifest.parent.glob("*.tmp")), [])

    def test_log_continues_when_manifest_is_locked(self):
        job = {"id": "b" * 32, "logs": []}
        with patch.dict(ui_server.JOBS, {job["id"]: job}), patch.object(
            ui_server, "write_manifest", side_effect=PermissionError(5, "Access is denied")
        ):
            ui_server.append_log(job["id"], "Recovering 3D motion")
        self.assertEqual(job["logs"], ["Recovering 3D motion"])


if __name__ == "__main__":
    unittest.main()
