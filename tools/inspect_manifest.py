"""
Scopo: strumento diagnostico da eseguire subito dopo preprocessing.py.
Mostra per ogni video la distribuzione crisi/non-crisi e la durata
effettiva della crisi in secondi, così puoi verificare visivamente
che le label corrispondano a quello che ti aspetti dal CSV.
Non fa parte della pipeline di training — eseguilo solo come controllo.
"""

import json
import collections
from pathlib import Path

MANIFEST_PATH = Path('data/manifest.json')

with open(MANIFEST_PATH) as f:
    records = json.load(f)

per_video = collections.defaultdict(lambda: {
    'total': 0, 'crisis': 0, 'onset': 0, 'offset': 0
})

for r in records:
    v = r['clip_name']                      
    per_video[v]['total']  += 1
    per_video[v]['crisis'] += r['label']
    per_video[v]['onset']   = r['onset_frame']
    per_video[v]['offset']  = r['offset_frame']

print(f"\n{'Clip':<40} {'Frames':>6} {'Crisi':>6} {'%':>6}  {'Durata crisi':>13}")
print("-" * 80)

for v, s in sorted(per_video.items()):
    pct          = 100 * s['crisis'] / s['total']
    duration_sec = (s['offset'] - s['onset']) / 30.0
    print(f"{v:<40} {s['total']:>6} {s['crisis']:>6} {pct:>5.1f}%  "
          f"{duration_sec:>10.1f}s")

total    = len(records)
n_crisis = sum(r['label'] == 1 for r in records)
n_normal = total - n_crisis

print("-" * 80)
print(f"{'TOTALE':<40} {total:>6} {n_crisis:>6} {100*n_crisis/total:>5.1f}%")
print(f"\npos_weight per BCEWithLogitsLoss → {n_normal/n_crisis:.4f}")

"""Output:
francescozanovello@Mac Tesi % python3 tools/inspect_manifest.py

Clip                                     Frames  Crisi      %   Durata crisi
--------------------------------------------------------------------------------
riga0_P95.mp4                               570    371  65.1%        37.0s
riga100_69.mp4                              430    231  53.7%        23.0s
riga101_69.mp4                              820    621  75.7%        62.0s
riga10_78.mp4                               480    281  58.5%        28.0s
riga11_78.mp4                               560    361  64.5%        36.0s
riga12_78.mp4                               540    341  63.1%        34.0s
riga13_83.mp4                               660    461  69.8%        46.0s
riga14_83.mp4                               650    451  69.4%        45.0s
riga15_83.mp4                               750    551  73.5%        55.0s
riga16_83.mp4                               810    611  75.4%        61.0s
riga17_83.mp4                               700    501  71.6%        50.0s
riga18_83.mp4                               820    621  75.7%        62.0s
riga19_83.mp4                               690    491  71.2%        49.0s
riga1_P95.mp4                               490    291  59.4%        29.0s
riga20_83.mp4                               510    311  61.0%        31.0s
riga21_83.mp4                               430    231  53.7%        23.0s
riga22_83.mp4                               570    371  65.1%        37.0s
riga23_83.mp4                               530    331  62.5%        33.0s
riga24_A4.mp4                               680    481  70.7%        48.0s
riga25_P95.mp4                              690    491  71.2%        49.0s
riga26_78.mp4                               630    431  68.4%        43.0s
riga27_78.mp4                               590    391  66.3%        39.0s
riga28_78.mp4                               600    401  66.8%        40.0s
riga29_78.mp4                               640    441  68.9%        44.0s
riga2_P95.mp4                               720    521  72.4%        52.0s
riga30_83.mp4                               540    341  63.1%        34.0s
riga31_83.mp4                               590    391  66.3%        39.0s
riga32_83.mp4                               590    391  66.3%        39.0s
riga33_83.mp4                               670    471  70.3%        47.0s
riga34_83.mp4                               550    351  63.8%        35.0s
riga35_83.mp4                               570    371  65.1%        37.0s
riga36_83.mp4                               530    331  62.5%        33.0s
riga37_83.mp4                               550    351  63.8%        35.0s
riga38_83.mp4                               450    251  55.8%        25.0s
riga39_83.mp4                               490    291  59.4%        29.0s
riga3_P95.mp4                               590    391  66.3%        39.0s
riga40_83.mp4                               650    451  69.4%        45.0s
riga41_83.mp4                               640    441  68.9%        44.0s
riga42_83.mp4                               550    351  63.8%        35.0s
riga43_83.mp4                               790    591  74.8%        59.0s
riga44_83.mp4                               640    441  68.9%        44.0s
riga45_83.mp4                               570    371  65.1%        37.0s
riga46_83.mp4                               720    521  72.4%        52.0s
riga47_83.mp4                               540    341  63.1%        34.0s
riga48_83.mp4                               640    441  68.9%        44.0s
riga49_83.mp4                               610    411  67.4%        41.0s
riga4_78.mp4                                520    321  61.7%        32.0s
riga50_83.mp4                               670    471  70.3%        47.0s
riga51_A1.mp4                              1010    811  80.3%        81.0s
riga53_A1.mp4                               710    511  72.0%        51.0s
riga54_A1.mp4                               650    451  69.4%        45.0s
riga55_A1.mp4                               630    431  68.4%        43.0s
riga56_A1.mp4                               560    361  64.5%        36.0s
riga57_A1.mp4                               630    431  68.4%        43.0s
riga58_A1.mp4                               610    411  67.4%        41.0s
riga59_A1.mp4                               710    511  72.0%        51.0s
riga5_78.mp4                                510    311  61.0%        31.0s
riga60_A1.mp4                               630    431  68.4%        43.0s
riga61_A1.mp4                               660    461  69.8%        46.0s
riga62_A1.mp4                               730    531  72.7%        53.0s
riga63_A1.mp4                               750    551  73.5%        55.0s
riga64_A1.mp4                               800    601  75.1%        60.0s
riga65_A4.mp4                               620    421  67.9%        42.0s
riga66_A4.mp4                               770    571  74.2%        57.0s
riga67_A4.mp4                               900    701  77.9%        70.0s
riga68_P95.mp4                             1200   1001  83.4%       100.0s
riga69_A7.mp4                               840    641  76.3%        64.0s
riga6_78.mp4                                470    271  57.7%        27.0s
riga70_A7.mp4                               570    371  65.1%        37.0s
riga71_78.mp4                               610    411  67.4%        41.0s
riga72_78.mp4                               670    471  70.3%        47.0s
riga73_78.mp4                               800    601  75.1%        60.0s
riga74_78.mp4                               790    591  74.8%        59.0s
riga75_78.mp4                               730    531  72.7%        53.0s
riga76_78.mp4                               840    641  76.3%        64.0s
riga77_78.mp4                               890    691  77.6%        69.0s
riga78_78.mp4                               670    471  70.3%        47.0s
riga79_78.mp4                               650    451  69.4%        45.0s
riga7_78.mp4                                460    261  56.7%        26.0s
riga80_78.mp4                               660    461  69.8%        46.0s
riga81_78.mp4                               730    531  72.7%        53.0s
riga82_78.mp4                               610    411  67.4%        41.0s
riga83_78.mp4                               557    381  68.4%        38.0s
riga85_A7.mp4                               490    291  59.4%        29.0s
riga86_78.mp4                              1220   1021  83.7%       102.0s
riga87_78.mp4                               900    701  77.9%        70.0s
riga88_78.mp4                               660    461  69.8%        46.0s
riga89_78.mp4                               810    611  75.4%        61.0s
riga8_78.mp4                                490    291  59.4%        29.0s
riga90_P95.mp4                              730    531  72.7%        53.0s
riga91_P95.mp4                              780    581  74.5%        58.0s
riga92_81.mp4                               670    471  70.3%        47.0s
riga93_81.mp4                               760    561  73.8%        56.0s
riga94_81.mp4                               660    461  69.8%        46.0s
riga95_81.mp4                               985    791  80.3%        79.0s
riga96_81.mp4                               550    351  63.8%        35.0s
riga97_69.mp4                               750    551  73.5%        55.0s
riga98_69.mp4                               650    451  69.4%        45.0s
riga99_69.mp4                               790    591  74.8%        59.0s
riga9_78.mp4                                720    521  72.4%        52.0s
--------------------------------------------------------------------------------
TOTALE                                    66462  46590  70.1%

pos_weight per BCEWithLogitsLoss → 0.4265
"""