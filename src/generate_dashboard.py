"""Generowanie interaktywnego panelu mapowego (HTML + Leaflet).

Tworzy dashboard_segmentacja.html wzorowany na dashboard_la_palma.html z Projektu 2:
- Podkład: Esri World Imagery, OpenStreetMap, Light Gray Canvas
- Warstwy: Sektory badawcze, Kataster INSPIRE 2026, Predykcje AI, Lawa 2021, BDOT10k Polska
- Przełącznik widoku: [Los Llanos, La Palma] / [Skawina, Polska]
- Panel statystyk i metryk transferu domeny (IoU, F1, Precision, Recall)
"""

import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box


def generate_dashboard(
    metrics_pl_path: Path = Path("outputs/poland_10cm/metrics.json"),
    metrics_zero_shot_path: Path = Path("outputs/los_llanos_10cm/metrics_zero_shot.json"),
    pred_es_gpkg: Path = Path("outputs/predictions_zero_shot_los_llanos_test.gpkg"),
    pred_pl_gpkg: Path = Path("outputs/predictions_poland_test.gpkg"),
    # Model fine-tuned z pełnymi 15 epokami (wersja z early stopping wybrała epokę 1 i jest zaniżona)
    metrics_finetuned_path: Path = Path("outputs/los_llanos_10cm_ft15/metrics.json"),
    pred_es_ft_gpkg: Path = Path("outputs/predictions_finetuned15_los_llanos_test.gpkg"),
    catastro_gpkg: Path = Path("data/los_llanos/buildings_los_llanos_2026.gpkg"),
    bdot_gpkg: Path = Path("data/poland/buildings_bdot10k_skawina.gpkg"),
    output_html: Path = Path("dashboard_segmentacja.html")
):
    print("Przygotowywanie danych do dashboardu...")

    # 1. Metryki
    m_pl = {}
    if metrics_pl_path.exists():
        with open(metrics_pl_path, "r", encoding="utf-8") as f:
            m_pl = json.load(f)

    m_es = {}
    if metrics_zero_shot_path.exists():
        with open(metrics_zero_shot_path, "r", encoding="utf-8") as f:
            m_es = json.load(f)

    m_ft = {}
    if metrics_finetuned_path.exists():
        with open(metrics_finetuned_path, "r", encoding="utf-8") as f:
            m_ft = json.load(f)

    # 2. Sektory Los Llanos (EPSG:32628 -> EPSG:4326)
    # Zaktualizowany sektor treningowy podniesiony o 200 m w górę: [214200, 3173600, 214600, 3174000]
    sectors_es_data = [
        {"name": "Sektor Treningowy (Train)", "role": "train", "geometry": box(214200, 3173600, 214600, 3174000)},
        {"name": "Sektor Walidacyjny (Val)", "role": "val", "geometry": box(216200, 3173400, 216600, 3173800)},
        {"name": "Sektor Testowy (Test)", "role": "test", "geometry": box(215400, 3173400, 215800, 3173800)},
    ]
    sectors_es_gdf = gpd.GeoDataFrame(sectors_es_data, crs="EPSG:32628").to_crs("EPSG:4326")
    sectors_es_geojson = json.loads(sectors_es_gdf.to_json())

    # 3. Sektory Polska (EPSG:2178 -> EPSG:4326)
    sectors_pl_data = [
        {"name": "Skawina Północ (Arkusz 08)", "role": "north", "geometry": box(7415200, 5538000, 7416800, 5539000)},
        {"name": "Skawina Południe (Arkusz 13)", "role": "south", "geometry": box(7415200, 5537000, 7416800, 5538000)},
    ]
    sectors_pl_gdf = gpd.GeoDataFrame(sectors_pl_data, crs="EPSG:2178").to_crs("EPSG:4326")
    sectors_pl_geojson = json.loads(sectors_pl_gdf.to_json())

    # 4. Kataster Los Llanos
    catastro_geojson = {"type": "FeatureCollection", "features": []}
    n_catastro = 0
    if catastro_gpkg.exists():
        c_gdf = gpd.read_file(catastro_gpkg)
        c_gdf = c_gdf[c_gdf.geometry.notna() & ~c_gdf.geometry.is_empty].to_crs("EPSG:4326")
        c_gdf["area_m2"] = c_gdf.geometry.to_crs("EPSG:32628").area.round(1)
        c_gdf["source"] = "Kataster INSPIRE 2026 (Hiszpania)"
        catastro_geojson = json.loads(c_gdf[["source", "area_m2", "geometry"]].to_json())
        n_catastro = len(c_gdf)

    # 5. Predykcje Los Llanos
    pred_es_geojson = {"type": "FeatureCollection", "features": []}
    n_pred_es = 0
    if pred_es_gpkg.exists():
        p_gdf = gpd.read_file(pred_es_gpkg)
        if not p_gdf.empty:
            p_gdf = p_gdf[p_gdf.geometry.notna() & ~p_gdf.geometry.is_empty].to_crs("EPSG:4326")
            p_gdf["area_m2"] = p_gdf["area"].round(1) if "area" in p_gdf else p_gdf.geometry.to_crs("EPSG:32628").area.round(1)
            p_gdf["source"] = "Predykcja AI (U-Net ResNet34 Zero-Shot)"
            pred_es_geojson = json.loads(p_gdf[["source", "area_m2", "geometry"]].to_json())
            n_pred_es = len(p_gdf)

    # 5b. Predykcje Los Llanos po fine-tuningu
    pred_es_ft_geojson = {"type": "FeatureCollection", "features": []}
    n_pred_es_ft = 0
    if pred_es_ft_gpkg.exists():
        f_gdf = gpd.read_file(pred_es_ft_gpkg)
        if not f_gdf.empty:
            f_gdf = f_gdf[f_gdf.geometry.notna() & ~f_gdf.geometry.is_empty].to_crs("EPSG:4326")
            f_gdf["area_m2"] = f_gdf["area"].round(1) if "area" in f_gdf else f_gdf.geometry.to_crs("EPSG:32628").area.round(1)
            f_gdf["source"] = "Predykcja AI (U-Net ResNet34 Fine-Tuned PL→ES)"
            pred_es_ft_geojson = json.loads(f_gdf[["source", "area_m2", "geometry"]].to_json())
            n_pred_es_ft = len(f_gdf)

    # 6. Predykcje Polska & BDOT10k
    pred_pl_geojson = {"type": "FeatureCollection", "features": []}
    n_pred_pl = 0
    if pred_pl_gpkg.exists():
        p_pl_gdf = gpd.read_file(pred_pl_gpkg)
        if not p_pl_gdf.empty:
            p_pl_gdf = p_pl_gdf[p_pl_gdf.geometry.notna() & ~p_pl_gdf.geometry.is_empty].to_crs("EPSG:4326")
            p_pl_gdf["area_m2"] = p_pl_gdf["area"].round(1) if "area" in p_pl_gdf else p_pl_gdf.geometry.to_crs("EPSG:2178").area.round(1)
            p_pl_gdf["source"] = "Predykcja AI (Polska Baseline)"
            pred_pl_geojson = json.loads(p_pl_gdf[["source", "area_m2", "geometry"]].to_json())
            n_pred_pl = len(p_pl_gdf)

    bdot_pl_geojson = {"type": "FeatureCollection", "features": []}
    n_bdot_pl = 0
    if bdot_gpkg.exists():
        b_pl_gdf = gpd.read_file(bdot_gpkg)
        if not b_pl_gdf.empty:
            b_pl_gdf = b_pl_gdf[b_pl_gdf.geometry.notna() & ~b_pl_gdf.geometry.is_empty].to_crs("EPSG:4326")
            b_pl_gdf["area_m2"] = b_pl_gdf.geometry.to_crs("EPSG:2178").area.round(1)
            b_pl_gdf["source"] = "BDOT10k (GUGiK Referencja)"
            bdot_pl_geojson = json.loads(b_pl_gdf[["source", "area_m2", "geometry"]].to_json())
            n_bdot_pl = len(b_pl_gdf)

    # 7. Lawa 2021 (z Projektu 2)
    # Plik lawy z Projektu 2 (opcjonalny); szukamy w kilku miejscach
    lava_candidates = [Path("data/lava/EMSR546_lava_2021-12-17.geojson"),
                       Path("../Projekt grupowy 2/EGSM/lawa/EMSR546_lava_2021-12-17.geojson")]
    lava_path = next((p for p in lava_candidates if p.exists()), lava_candidates[0])
    lava_geojson = {"type": "FeatureCollection", "features": []}
    if lava_path.exists():
        try:
            lava_gdf = gpd.read_file(lava_path).to_crs("EPSG:4326")
            lava_gdf = lava_gdf[lava_gdf.geometry.notna()]
            lava_gdf["geometry"] = lava_gdf.geometry.simplify(0.0005)
            lava_geojson = json.loads(lava_gdf[["geometry"]].to_json())
        except Exception as e:
            print(f"Lawa geojson: {e}")

    # Statystyki do panelu bocznego
    iou_pl = f"{m_pl.get('iou', 0)*100:.1f}%" if m_pl else "—"
    f1_pl = f"{m_pl.get('f1', 0)*100:.1f}%" if m_pl else "—"
    rec_pl = f"{m_pl.get('recall', 0)*100:.1f}%" if m_pl else "—"
    prec_pl = f"{m_pl.get('precision', 0)*100:.1f}%" if m_pl else "—"

    iou_es = f"{m_es.get('iou', 0)*100:.1f}%" if m_es else "—"
    f1_es = f"{m_es.get('f1', 0)*100:.1f}%" if m_es else "—"
    rec_es = f"{m_es.get('recall', 0)*100:.1f}%" if m_es else "—"
    prec_es = f"{m_es.get('precision', 0)*100:.1f}%" if m_es else "—"

    has_ft = bool(m_ft)
    iou_ft = f"{m_ft.get('iou', 0)*100:.1f}%" if m_ft else "—"
    f1_ft = f"{m_ft.get('f1', 0)*100:.1f}%" if m_ft else "—"
    rec_ft = f"{m_ft.get('recall', 0)*100:.1f}%" if m_ft else "—"
    prec_ft = f"{m_ft.get('precision', 0)*100:.1f}%" if m_ft else "—"
    # Karty statystyk dla Los Llanos: po fine-tuningu, jeśli są wyniki; inaczej zero-shot
    if has_ft:
        es_cards = (f'<div><b class="akcent">{n_catastro}</b><small>budynków referencyjnych (Kataster 2026)</small></div>'
                    f'<div><b class="pred">{n_pred_es_ft}</b><small>dachów wykrytych (Fine-Tuned AI)</small></div>'
                    f'<div><b>{f1_ft}</b><small>F1-score (Fine-Tuned; Zero-Shot {f1_es})</small></div>'
                    f'<div><b>{iou_ft}</b><small>IoU (Fine-Tuned; Zero-Shot {iou_es})</small></div>')
    else:
        es_cards = (f'<div><b class="akcent">{n_catastro}</b><small>budynków referencyjnych (Kataster 2026)</small></div>'
                    f'<div><b class="pred">{n_pred_es}</b><small>dachów wykrytych (Zero-Shot AI)</small></div>'
                    f'<div><b>{f1_es}</b><small>F1-score (Zero-Shot)</small></div>'
                    f'<div><b>{iou_es}</b><small>IoU (Jaccard)</small></div>')
    ft_legend = (f'<div class="wiersz"><i style="background: rgba(234, 179, 8, 0.45); border-color: #a16207;"></i>'
                 f'<span>Predykcje AI (Fine-Tuned ES) <b>{n_pred_es_ft}</b></span></div>') if has_ft else ""

    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="referrer" content="no-referrer">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Segmentacja Dachów Budynków · Polska vs Los Llanos de Aridane</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {{
      --tlo: #f4f5f7; --karta: #ffffff; --ramka: #e3e6ea; --tekst: #111827; --tekst-2: #4b5563;
      --wyciszony: #6b7280; --akcent: #b2182b; --sukces: #10b981; --predykcja: #f97316;
      --promien: 10px;
      --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; width: 100%; background: var(--tlo); color: var(--tekst); font: 14px/1.5 var(--font); overflow: hidden; }}
    
    .topbar {{
      position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding: 0 24px; background: var(--karta); border-bottom: 1px solid var(--ramka);
    }}
    .marka {{ display: flex; align-items: center; gap: 12px; }}
    .znak {{
      width: 32px; height: 32px; border-radius: 8px; background: #2563eb;
      display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 16px;
    }}
    .marka h1 {{ font-size: 15px; font-weight: 600; line-height: 1.2; }}
    .marka p {{ font-size: 12px; color: var(--wyciszony); }}
    
    .przyciski-skok {{ display: flex; gap: 8px; }}
    .btn-skok {{
      display: inline-flex; align-items: center; gap: 6px; padding: 7px 14px;
      border: 1px solid var(--ramka); border-radius: 8px; background: #fff;
      font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s;
    }}
    .btn-skok:hover {{ background: #f3f4f6; border-color: #d1d5db; }}
    .btn-skok.aktywny {{ background: #eff6ff; border-color: #3b82f6; color: #1d4ed8; box-shadow: 0 1px 3px rgba(37,99,235,.2); }}

    #map {{
      position: absolute; top: 64px; left: 0; right: 360px; bottom: 0;
      background: #e5e7eb;
    }}

    aside.panel {{
      position: fixed; top: 64px; right: 0; bottom: 0; width: 360px; z-index: 900;
      padding: 20px; overflow-y: auto; background: var(--karta);
      border-left: 1px solid var(--ramka);
    }}
    .etykieta {{
      display: block; margin: 0 0 8px; color: var(--wyciszony);
      font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    }}
    .sekcja {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--ramka); }}
    .liczby {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .liczby div {{ padding: 10px 12px; border: 1px solid var(--ramka); border-radius: 8px; background: #fafafa; }}
    .liczby b {{ display: block; font-size: 20px; font-weight: 700; line-height: 1.1; color: var(--tekst); }}
    .liczby b.akcent {{ color: #2563eb; }}
    .liczby b.pred {{ color: var(--predykcja); }}
    .liczby small {{ display: block; margin-top: 3px; color: var(--wyciszony); font-size: 11.5px; line-height: 1.3; }}
    
    .wiersz {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }}
    .wiersz i {{ flex: none; width: 14px; height: 14px; border: 1.5px solid rgba(0,0,0,.2); border-radius: 3px; }}
    .wiersz span {{ flex: 1; display: flex; justify-content: space-between; }}
    .wiersz span b {{ font-weight: 600; }}
    
    .opis {{ margin: 6px 0 0; color: var(--tekst-2); font-size: 12px; line-height: 1.55; }}
    .tabela-metryk {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }}
    .tabela-metryk th, .tabela-metryk td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--ramka); }}
    .tabela-metryk th {{ color: var(--wyciszony); font-weight: 600; }}
    
    .leaflet-popup-content-wrapper {{ border-radius: var(--promien); box-shadow: 0 4px 18px rgba(17,24,39,.14); }}
    .leaflet-popup-content {{ margin: 12px 14px; font-size: 12.5px; line-height: 1.5; }}

    @media (max-width: 900px) {{
      #map {{ right: 0; bottom: 45vh; }}
      aside.panel {{ top: auto; left: 0; right: 0; bottom: 0; width: 100%; height: 45vh; border-left: none; border-top: 1px solid var(--ramka); }}
    }}
  </style>
</head>
<body>

<header class="topbar">
  <div class="marka">
    <div class="znak">AI</div>
    <div>
      <h1>Detekcja Dachów Budynków z Ortofotomapy</h1>
      <p>Transfer domeny: Polska (BDOT10k) ➔ Los Llanos de Aridane (GRAFCAN 2024 / Kataster)</p>
    </div>
  </div>
  <div class="przyciski-skok">
    <button class="btn-skok aktywny" id="btn-es" onclick="skoczDoES()">📍 Los Llanos (La Palma)</button>
    <button class="btn-skok" id="btn-pl" onclick="skoczDoPL()">🇵🇱 Skawina (Polska)</button>
  </div>
</header>

<div id="map"></div>

<aside class="panel">
  <span class="etykieta" id="stat-title">Sektor Testowy · Los Llanos</span>
  <div class="liczby" id="stat-cards">
    {es_cards}
  </div>

  <div class="sekcja">
    <span class="etykieta">Warstwy i Legenda</span>
    <div class="wiersz"><i style="background: rgba(249, 115, 22, 0.4); border-color: #ea580c;"></i><span>Predykcje AI (Zero-Shot ES) <b>{n_pred_es}</b></span></div>
    {ft_legend}
    <div class="wiersz"><i style="background: rgba(16, 185, 129, 0.25); border-color: #059669;"></i><span>Kataster referencyjny INSPIRE <b>{n_catastro}</b></span></div>
    <div class="wiersz"><i style="background: rgba(37, 99, 235, 0.4); border-color: #1d4ed8;"></i><span>Predykcje AI (Polska Baseline) <b>{n_pred_pl}</b></span></div>
    <div class="wiersz"><i style="background: rgba(139, 92, 246, 0.25); border-color: #7c3aed;"></i><span>Referencja BDOT10k (Polska) <b>{n_bdot_pl}</b></span></div>
    <div class="wiersz"><i style="background: transparent; border: 2px dashed #3b82f6;"></i><span>Sektory badawcze (400×400 m)</span></div>
    <div class="wiersz"><i style="background: rgba(154, 52, 18, 0.4); border-color: #431407;"></i><span>Pole lawy 2021 (Copernicus)</span></div>
  </div>

  <div class="sekcja">
    <span class="etykieta">Porównanie Ewaluacji (Test Splits)</span>
    <table class="tabela-metryk">
      <thead>
        <tr><th>Metryka</th><th>PL ➔ PL (Baseline)</th><th>PL ➔ ES (Zero-Shot)</th><th>PL ➔ ES (Fine-Tuning)</th></tr>
      </thead>
      <tbody>
        <tr><td><b>IoU</b></td><td><b>{iou_pl}</b></td><td><b>{iou_es}</b></td><td><b>{iou_ft}</b></td></tr>
        <tr><td><b>F1-Score</b></td><td>{f1_pl}</td><td>{f1_es}</td><td>{f1_ft}</td></tr>
        <tr><td><b>Precision</b></td><td>{prec_pl}</td><td>{prec_es}</td><td>{prec_ft}</td></tr>
        <tr><td><b>Recall</b></td><td>{rec_pl}</td><td>{rec_es}</td><td>{rec_ft}</td></tr>
      </tbody>
    </table>
    <p class="opis" style="margin-top: 8px;">
      <b>Transfer domeny:</b> Model wytrenowany na polskich dachówkach ceramicznych i dachach skośnych musi zmierzyć się z płaskimi dachami (tarasy <i>azoteas</i>), ciemnym gruntem wulkanicznym i białą zabudową Kanarów.
    </p>
  </div>

  <div class="sekcja">
    <span class="etykieta">Weryfikacja w QGIS</span>
    <p class="opis">
      Wszystkie predykcje i warstwy są wyeksportowane jako standardowe pliki <b>GeoPackage (.gpkg)</b>:
    </p>
    <p class="opis" style="font-family: monospace; font-size: 11px; margin-top: 4px; background: #f3f4f6; padding: 6px; border-radius: 4px;">
      outputs/predictions_zero_shot_los_llanos_test.gpkg<br>
      outputs/predictions_finetuned15_los_llanos_test.gpkg<br>
      outputs/predictions_poland_test.gpkg<br>
      data/los_llanos/buildings_los_llanos_2026.gpkg<br>
      data/poland/buildings_bdot10k_skawina.gpkg
    </p>
  </div>
</aside>

<script>
  // Inicjalizacja mapy Leaflet
  const map = L.map('map', {{
    center: [28.6585, -17.9150],
    zoom: 16,
    maxZoom: 20
  }});

  // Podkłady kafelkowe (bez błędu 403)
  const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Esri World Imagery', maxZoom: 19
  }}).addTo(map);

  const cartoVoyager = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19, subdomains: 'abcd'
  }});

  const esriStreets = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Esri World Street Map', maxZoom: 19
  }});

  const esriLight = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Esri Light Gray Canvas', maxZoom: 19
  }});

  // Dane GeoJSON
  const sectorsEsGeo = {json.dumps(sectors_es_geojson)};
  const sectorsPlGeo = {json.dumps(sectors_pl_geojson)};
  const catastroGeo = {json.dumps(catastro_geojson)};
  const predEsGeo = {json.dumps(pred_es_geojson)};
  const predEsFtGeo = {json.dumps(pred_es_ft_geojson)};
  const predPlGeo = {json.dumps(pred_pl_geojson)};
  const bdotPlGeo = {json.dumps(bdot_pl_geojson)};
  const lavaGeo = {json.dumps(lava_geojson)};

  // Style
  function styleSector(f) {{
    const role = f.properties.role;
    const col = role === 'train' ? '#2563eb' : (role === 'val' ? '#7c3aed' : '#db2777');
    return {{ color: col, weight: 2.5, dashArray: '6, 6', fillOpacity: 0.06, fillColor: col }};
  }}

  function styleCatastro(f) {{
    return {{ color: '#059669', weight: 1.2, fillColor: '#10b981', fillOpacity: 0.3 }};
  }}

  function stylePredEs(f) {{
    return {{ color: '#c2410c', weight: 1.5, fillColor: '#f97316', fillOpacity: 0.55 }};
  }}

  function stylePredPl(f) {{
    return {{ color: '#1d4ed8', weight: 1.5, fillColor: '#3b82f6', fillOpacity: 0.55 }};
  }}

  function styleBdotPl(f) {{
    return {{ color: '#6d28d9', weight: 1.2, fillColor: '#8b5cf6', fillOpacity: 0.3 }};
  }}

  function styleLava(f) {{
    return {{ color: '#431407', weight: 1.5, fillColor: '#9a3412', fillOpacity: 0.35 }};
  }}

  function onEachPoly(f, layer) {{
    let p = f.properties || {{}};
    let txt = `<b>${{p.source || 'Budynek'}}</b><br>`;
    if (p.name) txt += `Nazwa: ${{p.name}}<br>`;
    if (p.area_m2) txt += `Powierzchnia dachu: <b>${{p.area_m2}} m²</b><br>`;
    layer.bindPopup(txt);
  }}

  // Warstwy Hiszpania (Los Llanos)
  const layerSectorsEs = L.geoJSON(sectorsEsGeo, {{
    style: styleSector,
    onEachFeature: (f, l) => l.bindTooltip(`<b>${{f.properties.name}}</b>`, {{ permanent: true, direction: 'top' }})
  }}).addTo(map);

  const layerCatastro = L.geoJSON(catastroGeo, {{ style: styleCatastro, onEachFeature: onEachPoly }}).addTo(map);
  const layerPredEs = L.geoJSON(predEsGeo, {{ style: stylePredEs, onEachFeature: onEachPoly }}).addTo(map);
  const layerPredEsFt = L.geoJSON(predEsFtGeo, {{
    style: () => ({{ color: '#a16207', weight: 1.2, fillColor: '#eab308', fillOpacity: 0.45 }}),
    onEachFeature: onEachPoly
  }});
  if (predEsFtGeo.features.length > 0) layerPredEsFt.addTo(map);
  const layerLava = L.geoJSON(lavaGeo, {{ style: styleLava }});
  if (lavaGeo.features.length > 0) layerLava.addTo(map);

  // Warstwy Polska (Skawina)
  const layerSectorsPl = L.geoJSON(sectorsPlGeo, {{
    style: styleSector,
    onEachFeature: (f, l) => l.bindTooltip(`<b>${{f.properties.name}}</b>`, {{ permanent: true, direction: 'top' }})
  }});
  const layerBdotPl = L.geoJSON(bdotPlGeo, {{ style: styleBdotPl, onEachFeature: onEachPoly }});
  const layerPredPl = L.geoJSON(predPlGeo, {{ style: stylePredPl, onEachFeature: onEachPoly }});

  // Kontroler warstw
  const baseMaps = {{
    "🛰️ Zdjęcia satelitarne (Esri)": esriSat,
    "🗺️ Mapa uliczna (CARTO Voyager)": cartoVoyager,
    "🧭 Mapa uliczna (Esri Streets)": esriStreets,
    "⚪ Podkład jasny (Esri Canvas)": esriLight
  }};

  const overlayMaps = {{
    "🟧 Predykcje AI (Zero-Shot ES)": layerPredEs,
    "🟨 Predykcje AI (Fine-Tuned ES)": layerPredEsFt,
    "🟩 Kataster INSPIRE (Hiszpania)": layerCatastro,
    "🔲 Sektory Los Llanos": layerSectorsEs,
    "🌋 Lawa 2021 (Copernicus)": layerLava,
    "🇵🇱 Sektory Polska (Skawina)": layerSectorsPl,
    "🟪 Referencja BDOT10k (Polska)": layerBdotPl,
    "🟦 Predykcje AI Polska": layerPredPl
  }};

  L.control.layers(baseMaps, overlayMaps, {{ collapsed: false }}).addTo(map);

  // Funkcje skoku
  function skoczDoES() {{
    document.getElementById('btn-es').classList.add('aktywny');
    document.getElementById('btn-pl').classList.remove('aktywny');
    
    // Włącz warstwy ES, wyłącz PL
    if (!map.hasLayer(layerSectorsEs)) layerSectorsEs.addTo(map);
    if (!map.hasLayer(layerCatastro)) layerCatastro.addTo(map);
    if (!map.hasLayer(layerPredEs)) layerPredEs.addTo(map);
    if (predEsFtGeo.features.length > 0 && !map.hasLayer(layerPredEsFt)) layerPredEsFt.addTo(map);
    if (map.hasLayer(layerSectorsPl)) map.removeLayer(layerSectorsPl);
    if (map.hasLayer(layerBdotPl)) map.removeLayer(layerBdotPl);
    if (map.hasLayer(layerPredPl)) map.removeLayer(layerPredPl);

    map.fitBounds(layerSectorsEs.getBounds(), {{ padding: [30, 30] }});

    document.getElementById('stat-title').innerText = "Sektor Testowy · Los Llanos (Hiszpania)";
    document.getElementById('stat-cards').innerHTML = `{es_cards}`;
  }}

  function skoczDoPL() {{
    document.getElementById('btn-pl').classList.add('aktywny');
    document.getElementById('btn-es').classList.remove('aktywny');
    
    // Włącz warstwy PL, wyłącz ES
    if (!map.hasLayer(layerSectorsPl)) layerSectorsPl.addTo(map);
    if (!map.hasLayer(layerBdotPl)) layerBdotPl.addTo(map);
    if (!map.hasLayer(layerPredPl)) layerPredPl.addTo(map);
    if (map.hasLayer(layerSectorsEs)) map.removeLayer(layerSectorsEs);
    if (map.hasLayer(layerCatastro)) map.removeLayer(layerCatastro);
    if (map.hasLayer(layerPredEs)) map.removeLayer(layerPredEs);
    if (map.hasLayer(layerPredEsFt)) map.removeLayer(layerPredEsFt);

    map.fitBounds(layerSectorsPl.getBounds(), {{ padding: [30, 30] }});

    document.getElementById('stat-title').innerText = "Obszar Testowy · Skawina (Polska)";
    document.getElementById('stat-cards').innerHTML = `
      <div><b class="akcent">{n_bdot_pl}</b><small>budynków referencyjnych (BDOT10k)</small></div>
      <div><b class="pred">{n_pred_pl}</b><small>dachów wykrytych (Baseline AI)</small></div>
      <div><b>{f1_pl}</b><small>F1-score (Baseline)</small></div>
      <div><b>{iou_pl}</b><small>IoU (Jaccard)</small></div>
    `;
  }}
</script>

</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Pomyślnie wygenerowano interaktywny dashboard: {output_html}")


if __name__ == "__main__":
    generate_dashboard()
