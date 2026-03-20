import pandas as pd
import os
import subprocess
import cv2
import csv

#CONFIGURAZIONE 
FILE_EXCEL = "File crisi.xlsx"
CARTELLA_VIDEO_INPUT = "Video/"       
CARTELLA_OUTPUT = "dataset_tagliato/"
BUFFER_SECONDI = 10

#Coordinate 210x210
W, H = 210, 210
nomi_topi = ["78", "83", "A1", "A4", "A5", "A3", "P95", "A7"]
coords = [
    (80, 120), (265, 120), (570, 120), (780, 120),  #Riga sopra
    (80, 325), (265, 325), (570, 325), (780, 325)   #Riga sotto
]
MAPPA_GABBIE = dict(zip(nomi_topi, coords))

def get_fps(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps if fps > 0 else 25.0

def hms_to_seconds(hms):
    try:
        parts = list(map(int, str(hms).strip().split(':')))
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: return parts[0] * 60 + parts[1]
        return float(hms)
    except: return 0.0

def processa_riga_specifica():
    if not os.path.exists(CARTELLA_OUTPUT): os.makedirs(CARTELLA_OUTPUT)
    
    df = pd.read_excel(FILE_EXCEL)
    #Pulizia nomi colonne per evitare errori di spazi
    df.columns = [str(c).strip() for c in df.columns]
    
    print(f"\n--- Elaborazione Riga Singola ---")
    riga_idx = int(input(f"Inserisci l'indice della riga da analizzare (0 - {len(df)-1}): "))
    
    if riga_idx not in df.index:
        print("Indice non valido.")
        return

    row = df.iloc[riga_idx]
    
    #LOGICA PERCORSO FILE
    nome_file_excel = str(row['Nome file']).strip()
    if not nome_file_excel.lower().endswith('.mp4'): 
        nome_file_excel += '.mp4'
    
    #Estraggo i primi 10 caratteri per la cartella
    cartella_data = nome_file_excel[:10]
    
    #Percorso
    path_in = os.path.join(CARTELLA_VIDEO_INPUT, cartella_data, nome_file_excel)
    
    if not os.path.exists(path_in):
        print(f"Errore: Il file non esiste in {path_in}")
        return

    #DATI TOPO E TEMPI
    id_topo = str(row['Animale']).strip()
    x, y = MAPPA_GABBIE.get(id_topo, (0,0))
    
    t_ini = hms_to_seconds(row['Inizio'])
    t_fine = hms_to_seconds(row['Fine'])
    
    start = max(0, t_ini - BUFFER_SECONDI)
    durata = (t_fine - t_ini) + (BUFFER_SECONDI * 2)
    fps = get_fps(path_in)
    
    output_name = f"riga{riga_idx}_{id_topo}.mp4"
    path_out = os.path.join(CARTELLA_OUTPUT, output_name)
    #mantengo questo qua per cambiare più velocemente coordinate di x e y
    #nomi_topi = ["78", "83", "A1", "A4", "A5", "A3", "P95", "A7"]
    #(80, 120), (265, 120), (570, 120), (780, 120),  # Riga sopra
    #(80, 325), (265, 325), (570, 325), (780, 325)   # Riga sotto
    #per 83:{210}:{100} #coordinate spostate immagini più piccole, non perfettamente inquadrato (devo mantenere 210x210)
    #varia in x -55, y -20 
    # FFmpeg: Taglio temporale + Crop 210x210 + No Audio
    comando = [
        'ffmpeg', '-y', '-ss', str(start), '-i', path_in,
        '-t', str(durata),
        '-vf', f"crop={W}:{H}:{145}:{280}",
        '-an', 
        '-c:v', 'libx264', '-crf', '18',
        path_out
    ]

    print(f"Elaborazione: {id_topo} | Cartella: {cartella_data} | File: {nome_file_excel}")
    subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Calcolo Frame per mappa_labels.csv
    off = t_ini - start
    f_ini = int(off * fps)
    f_fine = f_ini + int((t_fine - t_ini) * fps)

    #SCRITTURA LABEL
    file_label_path = os.path.join(CARTELLA_OUTPUT, "mappa_labels.csv")
    file_exists = os.path.isfile(file_label_path)
    
    #Uso newline='' per evitare problemi di righe fantasma o mancate
    with open(file_label_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        
        if not file_exists:
            writer.writerow(['clip_name', 'topo', 'fps', 'f_inizio', 'f_fine', 'f_tot'])
        
        #Scrivo i dati della clip
        writer.writerow([
            output_name, 
            id_topo, 
            round(fps, 2), 
            f_ini, 
            f_fine, 
            int(durata * fps)
        ])

    print(f"Completato! Clip salvata in {CARTELLA_OUTPUT}{output_name}")
    print(f"Dati registrati in: {file_label_path}")
    
    #Opzionale: apro il video subito per controllo
    os.system(f"open '{path_out}'")

if __name__ == "__main__":
    processa_riga_specifica()
    
   