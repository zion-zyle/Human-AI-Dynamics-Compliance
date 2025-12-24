import os
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from simulator import Simulator

# ==============================================================================
# [API Key Load] main.py 파일 위치 기준으로 env/.env를 읽어 OPENAI_API_KEY 등록
# ==============================================================================
if not os.getenv("OPENAI_API_KEY"):
    here = Path(__file__).resolve().parent  # ✅ main.py가 있는 폴더
    for env_file in ["env", ".env"]:
        p = here / env_file
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("OPENAI_API_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["OPENAI_API_KEY"] = key
                        print(f"[System] Loaded API Key from {p}")
                        break

# ==============================================================================
# [Configuration]
# ==============================================================================
DEFAULT_CONFIG = {
    "n_sessions": 55,
    "turns_per_session": 10,
    "seed": 42,
    "goal": 4.0,
    "out_root": "outputs",
    "model_name": "gpt-5-nano",
}

# ==============================================================================
# [User Profiles] BED 시뮬레이션을 위한 파라미터 세팅
# ==============================================================================
USER_PROFILES = [
    {
        "name": "independent_user",
        "age": 28, "gender": "M", "condition": "BED",
        "mu": 2.6, "beta": 0.01, "alpha": 0.02, "noise_std": 0.75,
        "compliance_max": 0.4, "tolerance": 1.2,
    },
    {
        "name": "compliant_user",
        "age": 31, "gender": "F", "condition": "BED",
        "mu": 2.4, "beta": 0.50, "alpha": 0.03, "noise_std": 0.40,
        "compliance_max": 1.0, "tolerance": 1.5,
    },
    {
        "name": "adaptive_user",
        "age": 35, "gender": "F", "condition": "BED",
        "mu": 2.5, "beta": 0.45, "alpha": 0.15, "noise_std": 0.30,
        "compliance_max": 1.0, "tolerance": 1.2,
    },
    {
        "name": "high_noise_user",
        "age": 26, "gender": "M", "condition": "BED",
        "mu": 2.7, "beta": 0.30, "alpha": 0.05, "noise_std": 1.20,
        "compliance_max": 0.9, "tolerance": 1.2,
    },
    {
        "name": "resistant_user",
        "age": 33, "gender": "M", "condition": "BED",
        "mu": 2.6, "beta": 0.30, "alpha": 0.10, "noise_std": 0.30,
        "compliance_max": 1.0, "tolerance": 0.55,
    },
]

def _safe_rmtree(p: Path) -> None:
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)

def run_one(profile: dict, config: dict) -> str:
    out_root = Path(config["out_root"])
    out_dir = out_root / profile["name"]
    _safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_config.json").write_text(
        json.dumps({"config": config, "profile": profile}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sim = Simulator(
        user_profile=profile,
        goal=float(config["goal"]),
        n_sessions=int(config["n_sessions"]),
        turns_per_session=int(config["turns_per_session"]),
        seed=int(config["seed"]),
        out_dir=str(out_dir),
        model_name=str(config["model_name"]),
    )
    sim.train()
    return str(out_dir)

def main():
    config = dict(DEFAULT_CONFIG)
    out_root = Path(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting Simulation with Model: {config['model_name']}")

    created = []
    with ProcessPoolExecutor(max_workers=min(5, os.cpu_count() or 2)) as ex:
        futs = [ex.submit(run_one, profile, config) for profile in USER_PROFILES]
        for fut in as_completed(futs):
            created.append(fut.result())

    print("All sessions finished.")
    for d in sorted(created):
        print(" - Output:", d)

if __name__ == "__main__":
    main()
