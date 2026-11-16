"""Uruchamia eksperymenty Mikołaja (M.1–M.5) po kolei.

Użycie (z katalogu bdot10k_bldg_segment_new, w aktywnym środowisku):
    python run_experiments.py                 # M.1, M.2, M.3 + tabela (wymagane)
    python run_experiments.py m1              # tylko fine-tuning
    python run_experiments.py m1 m3 summary   # wybrane kroki
    python run_experiments.py all             # wszystko, łącznie z opcjonalnym M.4 (enkodery)
    python run_experiments.py m2 --force      # policz od nowa, nawet jeśli wyniki istnieją

Kroki, których wynik już istnieje, są pomijane — można przerwać (Ctrl+C) i wznowić.
Logi każdego polecenia: outputs/logs/*.log
"""
import os, subprocess, sys, time
from pathlib import Path

PY = sys.executable
FORCE = "--force" in sys.argv
LOGS = Path("outputs/logs"); LOGS.mkdir(parents=True, exist_ok=True)

def run(name, args, done_if=None):
    if done_if and Path(done_if).exists() and not FORCE:
        print(f"[pomijam] {name} — istnieje {done_if}")
        return
    cmd = [PY, "-m"] + args
    print(f"\n=== {name}\n    {' '.join(cmd)}")
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}  # polskie znaki na Windows
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            try: sys.stdout.write(line)
            except UnicodeEncodeError: sys.stdout.write(line.encode("ascii", "replace").decode())
            log.write(line); log.flush()
        p.wait()
    if p.returncode != 0:
        sys.exit(f"\n!!! Krok '{name}' zakończył się błędem (kod {p.returncode}). Zobacz outputs/logs/{name}.log")
    print(f"    ok ({(time.time()-t0)/60:.1f} min)")

def tiles_ready(cfg_out):
    return Path(cfg_out, "tiles", "images").exists() and any(Path(cfg_out, "tiles", "images").glob("*.npy"))

def prep(name, config, out_dir):
    if tiles_ready(out_dir) and not FORCE:
        print(f"[pomijam] {name} — kafle już są w {out_dir}/tiles"); return
    run(name, ["src.prepare_data", "--config", config])

def m1():
    prep("m1_prepare_los_llanos", "configs/config_los_llanos_10cm.yaml", "outputs/los_llanos_10cm")
    # Fine-tuning: start z wag PL (init_checkpoint w configu)
    run("m1_train_finetune", ["src.train", "--config", "configs/config_los_llanos_10cm.yaml"],
        done_if="outputs/los_llanos_10cm/checkpoints/best.pt")
    run("m1_eval_finetune", ["src.evaluate", "--config", "configs/config_los_llanos_10cm.yaml",
        "--output", "outputs/los_llanos_10cm/metrics_finetuned.json"], done_if="outputs/los_llanos_10cm/metrics_finetuned.json")
    run("m1_predict_finetune", ["src.predict", "--config", "configs/config_los_llanos_10cm.yaml",
        "--input", "data/los_llanos/orthophoto_test_2024_10cm.tif",
        "--output", "outputs/predictions_finetuned_los_llanos_test.gpkg"], done_if="outputs/predictions_finetuned_los_llanos_test.gpkg")
    # Kontrola: ten sam trening, ale bez wag z Polski
    run("m1_train_scratch", ["src.train", "--config", "configs/config_los_llanos_10cm_scratch.yaml"],
        done_if="outputs/los_llanos_10cm_scratch/checkpoints/best.pt")
    run("m1_eval_scratch", ["src.evaluate", "--config", "configs/config_los_llanos_10cm_scratch.yaml"],
        done_if="outputs/los_llanos_10cm_scratch/metrics.json")
    # Fine-tuning bez early stopping (ten sam budżet 15 epok co kontrola)
    run("m1_train_ft15", ["src.train", "--config", "configs/config_los_llanos_10cm_ft15.yaml"],
        done_if="outputs/los_llanos_10cm_ft15/checkpoints/best.pt")
    run("m1_eval_ft15", ["src.evaluate", "--config", "configs/config_los_llanos_10cm_ft15.yaml"],
        done_if="outputs/los_llanos_10cm_ft15/metrics.json")
    run("m1_eval_ft15_last", ["src.evaluate", "--config", "configs/config_los_llanos_10cm_ft15.yaml",
        "--checkpoint", "outputs/los_llanos_10cm_ft15/checkpoints/last.pt",
        "--output", "outputs/los_llanos_10cm_ft15/metrics_last_epoch.json"],
        done_if="outputs/los_llanos_10cm_ft15/metrics_last_epoch.json")
    run("m1_predict_ft15", ["src.predict", "--config", "configs/config_los_llanos_10cm_ft15.yaml",
        "--input", "data/los_llanos/orthophoto_test_2024_10cm.tif",
        "--output", "outputs/predictions_finetuned15_los_llanos_test.gpkg"],
        done_if="outputs/predictions_finetuned15_los_llanos_test.gpkg")

def m2():
    prep("m2_prepare_baseline", "configs/config.yaml", "outputs/poland_10cm")  # maski oryginalne do oceny
    for tag in ["ero000", "ero025", "ero050"]:
        cfg, out = f"configs/config_poland_10cm_{tag}.yaml", f"outputs/poland_10cm_{tag}"
        if tag != "ero000":  # ero000 korzysta z kafli baseline (splits_csv)
            prep(f"m2_prepare_{tag}", cfg, out)
        run(f"m2_train_{tag}", ["src.train", "--config", cfg], done_if=f"{out}/checkpoints/best.pt")
        # Ocena na NIEerodowanych maskach testowych (te same kafle testowe co baseline)
        run(f"m2_eval_{tag}", ["src.evaluate", "--config", "configs/config.yaml",
            "--checkpoint", f"{out}/checkpoints/best.pt", "--output", f"{out}/metrics_vs_original_masks.json"],
            done_if=f"{out}/metrics_vs_original_masks.json")

def m3():
    prep("m3_prepare_20cm", "configs/config_poland_20cm.yaml", "outputs/poland_20cm")
    run("m3_train_20cm", ["src.train", "--config", "configs/config_poland_20cm.yaml"], done_if="outputs/poland_20cm/checkpoints/best.pt")
    run("m3_eval_20cm", ["src.evaluate", "--config", "configs/config_poland_20cm.yaml"], done_if="outputs/poland_20cm/metrics.json")
    # Predykcje wektorowe na obu arkuszach w obu rozdzielczościach + metryki obiektowe (małe budynki)
    for res, cfg in [("10cm", "configs/config.yaml"), ("20cm", "configs/config_poland_20cm.yaml")]:
        preds = []
        for part in ["north", "south"]:
            out = f"outputs/poland_{res}/pred_{part}.gpkg"; preds.append(out)
            run(f"m3_predict_{res}_{part}", ["src.predict", "--config", cfg,
                "--input", f"data/poland/orthophoto_skawina_{part}_{res}.tif", "--output", out], done_if=out)
        run(f"m3_objects_{res}", ["src.object_metrics", "--config", cfg, "--predictions", *preds,
            "--output", f"outputs/poland_{res}/object_metrics.json"], done_if=f"outputs/poland_{res}/object_metrics.json")
    # Porównanie fair: model 10 cm trenowany tym samym protokołem (ero000, 15 epok),
    # oceniany na dokładnie tym samym obszarze co 20 cm (siatka 20 cm nie obejmuje skrajnych pasów ~50 m)
    preds = []
    for part in ["north", "south"]:
        out = f"outputs/poland_10cm_ero000/pred_{part}.gpkg"; preds.append(out)
        run(f"m3_predict_10cm_ero000_{part}", ["src.predict", "--config", "configs/config.yaml",
            "--checkpoint", "outputs/poland_10cm_ero000/checkpoints/best.pt",
            "--input", f"data/poland/orthophoto_skawina_{part}_10cm.tif", "--output", out], done_if=out)
    run("m3_objects_10cm_common", ["src.object_metrics", "--config", "configs/config_poland_20cm.yaml",
        "--predictions", *preds, "--output", "outputs/poland_10cm_ero000/object_metrics_common_area.json"],
        done_if="outputs/poland_10cm_ero000/object_metrics_common_area.json")

def m4():
    prep("m4_prepare_baseline", "configs/config.yaml", "outputs/poland_10cm")
    for tag in ["resnet50", "mit_b0"]:
        cfg, out = f"configs/config_poland_10cm_{tag}.yaml", f"outputs/poland_10cm_{tag}"
        run(f"m4_train_{tag}", ["src.train", "--config", cfg], done_if=f"{out}/checkpoints/best.pt")
        run(f"m4_eval_{tag}", ["src.evaluate", "--config", cfg], done_if=f"{out}/metrics.json")

def summary():
    run("m5_summary", ["src.make_summary"])

STEPS = {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "summary": summary}

if __name__ == "__main__":
    chosen = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    if not chosen: chosen = ["m1", "m2", "m3", "summary"]
    if "all" in chosen: chosen = ["m1", "m2", "m3", "m4", "summary"]
    if not Path("src").exists():
        sys.exit("Uruchom z katalogu bdot10k_bldg_segment_new")
    for c in chosen:
        if c not in STEPS: sys.exit(f"Nieznany krok: {c}. Dostępne: {', '.join(STEPS)} all")
        STEPS[c]()
    print("\nGotowe. Tabela: outputs/tabela_zbiorcza.md")
