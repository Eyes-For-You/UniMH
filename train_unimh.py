"""Train UniMH with dataset-specific MDS and PTR configurations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PRESETS = {
    "eth_ucy": ("fm.py", "cfg/eth_ucy/cor_fm.yml", "data/eth_ucy"),
    "sdd": ("fm.py", "cfg/sdd/cor_fm.yml", "data/sdd"),
    "giga": ("fm.py", "cfg/giga/cor_fm.yml", "data/giga"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the UniMH teacher with MDS and PTR enabled."
    )
    parser.add_argument("--dataset", required=True, choices=PRESETS)
    parser.add_argument(
        "--fold", choices=("eth", "hotel", "univ", "zara1", "zara2"),
        help="Leave-one-out fold; required for ETH-UCY only.",
    )
    parser.add_argument("--data-root", type=Path, help="Prepared dataset directory.")
    parser.add_argument("--config", type=Path, help="Override the dataset preset YAML.")
    parser.add_argument("--run-name", default="unimh", help="Experiment name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the resolved settings without loading data or starting training.",
    )
    args = parser.parse_args(argv)
    if args.dataset == "eth_ucy" and args.fold is None:
        parser.error("--fold is required for ETH-UCY")
    if args.dataset != "eth_ucy" and args.fold is not None:
        parser.error("--fold is supported only for ETH-UCY")
    for name in ("epochs", "batch_size", "learning_rate"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative")

    script, default_config, default_data = PRESETS[args.dataset]
    config_path = (args.config if args.config is not None else ROOT / default_config)
    config_path = config_path.expanduser().resolve()
    data_root = (args.data_root if args.data_root is not None else ROOT / default_data)
    data_root = data_root.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")

    import yaml

    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("dataset") != args.dataset:
        parser.error("The configuration dataset must match --dataset")
    moe = config.get("MODEL", {}).get("MOE", {})
    if not (moe.get("ENCODER_MOE", False) or moe.get("DECODER_MOE", False)):
        parser.error("The UniMH configuration must enable MDS expert routing")
    if not config.get("use_vqvae", False):
        parser.error("The UniMH configuration must enable the PTR training branch")

    # Preserve the teacher preprocessing, loss and optimization defaults.
    command = [
        sys.executable, str(ROOT / script), "--dataset", args.dataset,
        "--cfg", str(config_path), "--data_dir", str(data_root),
        "--rotate", "--rotate_time_frame", "6", "--tied_noise",
        "--fm_in_scaling", "--use_vqvae", "--fix_random_seed",
        "--seed", str(args.seed), "--checkpt_freq", "1",
        "--exp", args.run_name,
    ]
    if args.dataset == "eth_ucy":
        command += ["--subset", args.fold, "--data_source", "original"]
    elif args.dataset == "giga":
        command += ["--subset", "giga", "--data_source", "TUTR"]
    for value, option in [
        (args.epochs, "--epochs"), (args.batch_size, "--batch_size"),
        (args.learning_rate, "--init_lr"), (args.weight_decay, "--weight_decay"),
    ]:
        if value is not None:
            command += [option, str(value)]

    components = {
        "mds_encoder": bool(moe.get("ENCODER_MOE", False)),
        "mds_decoder": bool(moe.get("DECODER_MOE", False)),
        "ptr": True,
    }
    if args.dry_run:
        print(json.dumps({
            "dataset": args.dataset, "components": components,
            "training_histories": config["past_frames"],
            "training_horizons": config["future_frames"], "command": command,
        }, indent=2))
        return 0

    print(f"UniMH | {args.dataset} | {json.dumps(components)}", flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
