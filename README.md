# Segmentacja dachów budynków · Skawina ➔ La Palma
 
**Czy model segmentacji budynków wytrenowany w Polsce działa na Wyspach Kanaryjskich?**
Analiza transferu domeny między BDOT10k a katastrem w Los Llanos de Aridane.
 
🌐 **Strona z projektami:** https://student.agh.edu.pl/~jawilk/
🗺️ **Interaktywna mapa projektu:** https://student.agh.edu.pl/~jawilk/mapa4/
(serwer studencki AGH, strona działa w sieci uczelni lub przez VPN AGH)
 
Projekt grupowy 3 · *Przetwarzanie i analiza cyfrowych danych przestrzennych i budowlanych metodami sztucznej inteligencji* · AGH, semestr 7 (2026/27)
**Autorzy:** Jakub Wilk, Mikołaj Etrych, Krzysztof Nowakowski
 
---
 
## Pytanie badawcze
 
W jakim stopniu sieć U-Net wytrenowana na polskich ortofotomapach i obrysach BDOT10k przenosi się na zabudowę Los Llanos de Aridane (La Palma)? Czy dotrenowanie na małej próbce lokalnych danych zmniejsza błąd transferu domeny?
 
**Hipoteza:** model z Polski zachowa wysoką precyzję, ale będzie miał niski recall, bo nie zna płaskich dachów (*azoteas*) ani jasnej zabudowy Kanarów.
 
## Dane
 
| | Polska | Hiszpania |
|---|---|---|
| Obszar | Skawina / Rzozów, 2 arkusze (północ 08, południe 13) | Los Llanos de Aridane, 3 rozłączne sektory 400 × 400 m (train / val / test) |
| Obraz | Ortofotomapa GUGiK RGB, 2013, 10 cm/px (oraz wersja 20 cm/px) | Ortofotomapa GRAFCAN RGB, 2024 (WMS `ortoterritorial_3000`), 10 cm/px, EPSG:32628 |
| Etykiety | BDOT10k, budynki `OT_BUBD_A`, 1705 obiektów | Kataster INSPIRE (Dirección General del Catastro, gmina 38024), 826 obiektów |
 
Szczegółowe metadane każdego obrazu są w `METADANE.md`.
 
## Metoda
 
- **Model:** U-Net z enkoderem ResNet34 (wagi ImageNet), biblioteka `segmentation_models_pytorch`.
- **Strata:** 0,5 · BCE + 0,5 · Dice. Optymalizator AdamW, 15 epok, batch 8, próg 0,5.
- **Kafle:** 512 × 512 px, augmentacja (odbicia, obroty, jasność).
- **Podział bez wycieku danych:**
  - W Polsce kafle są grupowane w bloki 4 × 4 (204,8 m) i losowane jako całość: 70% / 15% / 15%.
  - W Hiszpanii każdy zbiór to osobny sektor miasta.
  - Wersja 20 cm używa dokładnie tych samych bloków geograficznych.
- **Predykcja:** kafle z zakładką (marginesy po 64 px), maska składana dla całego rastra, eksport poligonów do GeoPackage.
- **Metryki:**
  - pikselowe (IoU, F1, Precision, Recall) na zbiorach testowych;
  - obiektowe: budynek uznajemy za wykryty, gdy predykcja pokrywa ≥ 50% jego obrysu.
 
## Wyniki
 
Metryki pikselowe na zbiorach testowych (U-Net + ResNet34, 15 epok):
 
| Wariant | Rozdz. | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| PL → PL (baseline) | 10 cm | 65.3% | 79.0% | 78.0% | 80.1% |
| PL → ES (zero-shot) | 10 cm | 54.3% | 70.4% | 82.8% | 61.2% |
| **PL → ES (fine-tuning)** | 10 cm | **72.1%** | **83.8%** | **83.5%** | **84.1%** |
| ES → ES (kontrola, bez wag z Polski) | 10 cm | 71.2% | 83.2% | 82.9% | 83.4% |
| PL → PL, erozja masek 0,25 m | 10 cm | 65.7% | 79.3% | 79.9% | 78.6% |
| PL → PL, erozja masek 0,5 m | 10 cm | 65.9% | 79.4% | 83.3% | 75.9% |
| PL → PL | 20 cm | 66.3% | 79.7% | 77.6% | 82.0% |
 
Wykryte budynki (obszary testowe):
 
| | Wykryte |
|---|---|
| La Palma · zero-shot | 67.1% (243 / 362) |
| La Palma · po fine-tuningu | 95.0% (344 / 362) |
| Skawina · PL → PL | 79.3% (241 / 304) |
 
Wyników z Polski i Hiszpanii nie porównujemy wprost, bo etykiety pochodzą z różnych rejestrów. Pełna tabela jest w `outputs/tabela_zbiorcza.md`.
 
## Wnioski
 
1. **Hipoteza się potwierdziła.** Model z Polski działa na La Palmie z wysoką precyzją (82,8%), ale pomija około 40% powierzchni dachów (recall 61,2%).
2. **Dotrenowanie na 387 budynkach z La Palmy** podnosi IoU z 54,3% do 72,1% (+17,8 p.p.), a odsetek wykrytych budynków z 67% do 95%.
3. **Trening od zera na samych danych z La Palmy daje prawie to samo** (IoU 71,2%). Start z wag polskich dodaje tylko +0,9 p.p., co przy jednym przebiegu mieści się w szumie. O wyniku decydują lokalne dane.
4. **Erozja masek** nie zmienia IoU. Przesuwa za to kompromis: precision +5,3 p.p., recall −4,2 p.p.
5. **Rozdzielczość 20 cm** daje porównywalne IoU przy około 4× szybszej epoce (60 s wobec ok. 230 s). W tych danych nie gubi małych budynków.
 
## Ograniczenia
 
- Etykiety to obrysy budynków (przyziemia), a nie dachy, więc okapy liczą się jako błąd.
- Polska ortofotomapa jest z 2013 r., a BDOT10k jest aktualny.
- Każdy wariant ma jeden przebieg, więc różnice rzędu 1 p.p. to szum.
- Sektor walidacyjny na La Palmie (45 budynków na skraju miasta) jest mało reprezentatywny: walidacja daje ok. 46% IoU, test 72%.
- Bloki testowe w Polsce są losowane bez warunku na zabudowę. Jeden blok leży na rzece i obniża IoU o około 1 p.p.
 
## Struktura repozytorium
 
```
bdot10k_bldg_segment_new/
├── configs/              # YAML dla każdego eksperymentu (PL 10/20 cm, erozja, ES fine-tuning/kontrola)
├── data/
│   ├── poland/           # ortofotomapy Skawiny + BDOT10k (.gpkg)
│   └── los_llanos/       # ortofotomapy GRAFCAN + kataster INSPIRE (.gpkg)
├── src/
│   ├── prepare_data.py   # kafelkowanie, maski, podział przestrzenny
│   ├── train.py          # trening / fine-tuning (--init-checkpoint)
│   ├── evaluate.py       # metryki pikselowe na zbiorze testowym
│   ├── predict.py        # predykcja całego rastra -> GeoPackage
│   ├── object_metrics.py # metryki per budynek
│   ├── make_summary.py   # tabela zbiorcza
│   └── generate_dashboard.py + dashboard_template.html  # strona z mapą
├── outputs/              # checkpointy, metryki, predykcje .gpkg, logi
└── run_experiments.py    # uruchamia wszystkie eksperymenty po kolei
```
 
## Uruchomienie
 
```bash
cd bdot10k_bldg_segment_new
pip install -r requirements.txt
 
# wszystkie eksperymenty (gotowe kroki są pomijane, logi w outputs/logs/)
python run_experiments.py
 
# lub pojedynczo, np. baseline
python -m src.prepare_data --config configs/config.yaml
python -m src.train        --config configs/config.yaml
python -m src.evaluate     --config configs/config.yaml
python -m src.predict      --config configs/config.yaml \
  --input data/poland/orthophoto_skawina_south_10cm.tif --output outputs/pred.gpkg
 
# strona z mapą (dashboard_segmentacja.html)
python -m src.generate_dashboard
```
 
Kod obsługuje CUDA, Apple Silicon (MPS) i CPU.
 
## Interaktywna mapa
 
https://student.agh.edu.pl/~jawilk/mapa4/
 
- Przełącznik obszarów **La Palma ↔ Skawina** z animowanym przelotem nad Europą.
- Przełącznik modelu **Zero-Shot / po fine-tuningu** na tle obrysów katastru.
- Sektory i bloki testowe, pole lawy z erupcji 2021 (Copernicus EMSR546).
- Panel z metrykami pikselowymi, wykrytymi budynkami i wnioskami.
 
## Źródła danych
 
GUGiK (ortofotomapa, BDOT10k) · GRAFCAN (ortofotomapa 2024) · Dirección General del Catastro (INSPIRE BU) · Copernicus EMS (EMSR546) · Esri World Imagery
Szablon kodu: repozytorium z zajęć (P. Kuras, AGH).
