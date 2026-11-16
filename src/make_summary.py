"""Składa outputs/tabela_zbiorcza.md z plików metryk wszystkich eksperymentów (M.5).
Brakujące eksperymenty są oznaczane jako 'do uzup.' — można uruchamiać wielokrotnie.
"""
import json
from pathlib import Path
import pandas as pd

# (nazwa, rozdzielczość, enkoder, erozja, plik metryk, katalog z training_history.csv, uwagi)
EXPERIMENTS = [
    ("PL ➔ PL (Baseline)", "10 cm", "ResNet34", "0.0 m", "outputs/poland_10cm/metrics.json", None, "Model Jakuba — wagi z 3. epoki (test. przebieg)"),
    ("PL ➔ ES (Zero-Shot)", "10 cm", "ResNet34", "0.0 m", "outputs/los_llanos_10cm/metrics_zero_shot.json", None, "Czysty transfer domeny"),
    ("PL ➔ ES (Fine-Tuning)", "10 cm", "ResNet34", "0.0 m", "outputs/los_llanos_10cm/metrics_finetuned.json", "outputs/los_llanos_10cm", "Start z wag PL; early stopping wybrał epokę 1 (patience 6)"),
    ("PL ➔ ES (Fine-Tuning, 15 ep.)", "10 cm", "ResNet34", "0.0 m", "outputs/los_llanos_10cm_ft15/metrics.json", "outputs/los_llanos_10cm_ft15", "Start z wag PL, ten sam budżet epok co kontrola"),
    ("PL ➔ ES (Fine-Tuning, ostatnia epoka)", "10 cm", "ResNet34", "0.0 m", "outputs/los_llanos_10cm_ft15/metrics_last_epoch.json", None, "Wagi po 15. epoce, bez wyboru na walidacji"),
    ("ES ➔ ES (bez PL, kontrola)", "10 cm", "ResNet34", "0.0 m", "outputs/los_llanos_10cm_scratch/metrics.json", "outputs/los_llanos_10cm_scratch", "Te same kafle, start tylko z ImageNet"),
    ("PL ➔ PL (Erozja 0.0 m, 15 ep.)", "10 cm", "ResNet34", "0.0 m", "outputs/poland_10cm_ero000/metrics_vs_original_masks.json", "outputs/poland_10cm_ero000", "Punkt odniesienia dla erozji: ten sam protokół treningu"),
    ("PL ➔ PL (Erozja 0.25 m)", "10 cm", "ResNet34", "0.25 m", "outputs/poland_10cm_ero025/metrics_vs_original_masks.json", "outputs/poland_10cm_ero025", "Ocena na nieerodowanych maskach testowych"),
    ("PL ➔ PL (Erozja 0.5 m)", "10 cm", "ResNet34", "0.5 m", "outputs/poland_10cm_ero050/metrics_vs_original_masks.json", "outputs/poland_10cm_ero050", "Ocena na nieerodowanych maskach testowych"),
    ("PL ➔ PL (20 cm)", "20 cm", "ResNet34", "0.0 m", "outputs/poland_20cm/metrics.json", "outputs/poland_20cm", "Te same bloki testowe (siatka 20 cm bez skrajnych pasów)"),
    ("PL ➔ PL (ResNet50)", "10 cm", "ResNet50", "0.0 m", "outputs/poland_10cm_resnet50/metrics.json", "outputs/poland_10cm_resnet50", "Opcjonalnie"),
    ("PL ➔ PL (MiT-B0 / SegFormer)", "10 cm", "MiT-B0", "0.0 m", "outputs/poland_10cm_mit_b0/metrics.json", "outputs/poland_10cm_mit_b0", "Opcjonalnie"),
]

def pct(m, k):
    return f"{m[k]*100:.1f}%" if m and k in m else "*do uzup.*"

def epoch_time(d):
    if not d: return "—"
    p = Path(d) / "training_history.csv"
    if not p.exists(): return "—"
    h = pd.read_csv(p)
    if "epoch_time_s" not in h: return "—"
    return f"{h.epoch_time_s.mean():.0f} s ({len(h)} ep.)"

def load(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

def main():
    lines = ["# Tabela zbiorcza wyników (M.5)", "",
             "Metryki pikselowe na zbiorze testowym (próg 0.5). Wyników PL i ES nie porównujemy wprost — inne rejestry etykiet; "
             "kluczowe porównanie to Zero-Shot vs Fine-Tuning na tym samym hiszpańskim teście.", "",
             "| Eksperyment | Rozdzielczość | Enkoder | Erozja masek | IoU | F1-score | Precision | Recall | Śr. czas epoki | Uwagi |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, res, enc, ero, mpath, hdir, note in EXPERIMENTS:
        m = load(mpath)
        lines.append(f"| **{name}** | {res} | {enc} | {ero} | {pct(m,'iou')} | {pct(m,'f1')} | {pct(m,'precision')} | {pct(m,'recall')} | {epoch_time(hdir)} | {note} |")

    zs = load("outputs/los_llanos_10cm/metrics_zero_shot.json")
    ft = load("outputs/los_llanos_10cm_ft15/metrics.json") or load("outputs/los_llanos_10cm/metrics_finetuned.json")
    sc = load("outputs/los_llanos_10cm_scratch/metrics.json")
    if zs and ft:
        lines += ["", "## Efekt fine-tuningu (PL ➔ ES, sektor testowy Los Llanos, 15 epok)", "",
                  "| Metryka | Zero-Shot | Fine-Tuning PL➔ES | Zmiana vs Zero-Shot (p.p.) | ES➔ES bez PL | Fine-Tuning vs bez PL (p.p.) |",
                  "|---|---|---|---|---|---|"]
        for k, lab in [("iou","IoU"),("f1","F1"),("precision","Precision"),("recall","Recall")]:
            scv = f"{sc[k]*100:.1f}%" if sc else "—"
            scd = f"{(ft[k]-sc[k])*100:+.1f}" if sc else "—"
            lines.append(f"| {lab} | {zs[k]*100:.1f}% | {ft[k]*100:.1f}% | {(ft[k]-zs[k])*100:+.1f} | {scv} | {scd} |")

    obj = [("10 cm", "outputs/poland_10cm_ero000/object_metrics_common_area.json"), ("20 cm", "outputs/poland_20cm/object_metrics.json")]
    objs = [(r, load(p)) for r, p in obj]
    if any(o for _, o in objs):
        bins = next(o for _, o in objs if o)["recall_by_size"]
        lines += ["", "## Wykrywanie budynków wg wielkości (Polska, wspólny obszar testowy 10/20 cm, pokrycie ≥ 50%)", "",
                  "| Rozdzielczość | " + " | ".join(b["bin"] for b in bins) + " | Łącznie | Fałszywe poligony |",
                  "|---|" + "---|" * (len(bins) + 2)]
        for r, o in objs:
            if not o:
                lines.append(f"| {r} | " + " | ".join("*do uzup.*" for _ in bins) + " | *do uzup.* | *do uzup.* |"); continue
            cells = [f"{b['recall']*100:.0f}% ({b['n_detected']}/{b['n_reference']})" if b["recall"] is not None else "—" for b in o["recall_by_size"]]
            lines.append(f"| {r} | " + " | ".join(cells) + f" | {o['object_recall']*100:.0f}% | {o['n_false_polygons']} |")

    out = Path("outputs/tabela_zbiorcza.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
