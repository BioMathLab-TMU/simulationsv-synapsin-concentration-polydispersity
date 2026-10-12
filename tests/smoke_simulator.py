"""Four-step software smoke test; not a scientific production trajectory."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/SimulationSV"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    config = json.loads((ROOT / "configs/C10_mono_seed9210001.json").read_text(encoding="utf-8"))
    config["times"][0]["Ttime"] = 5
    config["total"][0]["load"] = 2
    config["total"][0]["address"] = "publication_software_smoke"
    with tempfile.TemporaryDirectory(prefix="simulationsv_public_smoke_") as workspace:
        folder = Path(workspace)
        short_config = folder / "smoke.json"
        short_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run([str(binary), str(short_config)], cwd=folder, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"Smoke test failed with {result.returncode}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        files = list((folder / "builds").glob("publication_software_smoke----*/simulation_Rawdata__*.h5"))
        if len(files) != 1 or files[0].stat().st_size == 0:
            raise AssertionError("Expected one nonempty HDF5 smoke output")
    print("SIMULATION_SOFTWARE_SMOKE_OK moves=4 production_claim=false")


if __name__ == "__main__":
    main()
