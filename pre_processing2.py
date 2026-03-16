#Codice per realizzare il pre-processing.
#Nel dettaglio questo codice prende tutti i video e in base al momento della crisi del file .xlsx
#estrai la relativa clip video con 10 secondi di offset (prima e dopo la crisi).
#Nel mentre crea pure un file mappa_labels.csv che scrive frame per frame se avviene la crisi o meno.
import pandas as pd
import os
import subprocess
import cv2
import csv


# 1. CONFIGURAZIONE PERCORSI E COLONNE
FILE_EXCEL = "File crisi.xlsx"     
CARTELLA_VIDEO_INPUT = "Video/"   
CARTELLA_OUTPUT = "dataset_tagliato/"       
BUFFER_SECONDI = 10                        

# Nomi esatti delle colonne
COLONNA_FILE = "Nome file"    
COLONNA_INIZIO = "Inizio" 
COLONNA_FINE = "Fine"

# Nome del file finale con le etichette per il cluster
FILE_LABELS = os.path.join(CARTELLA_OUTPUT, "mappa_labels.csv")


# 2. FUNZIONI DI SUPPORTO
def get_video_info(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps if fps > 0 else 25.0

def hms_to_seconds(hms):
    """Converte HH:MM:SS in secondi totali"""
    try:
        parts = list(map(int, str(hms).strip().split(':')))
        if len(parts) == 3: # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: # MM:SS
            return parts[0] * 60 + parts[1]
        return float(hms)
    except:
        return 0.0


# 3. PROCESSO DI ESTRAZIONE
if not os.path.exists(CARTELLA_OUTPUT):
    os.makedirs(CARTELLA_OUTPUT)

# Caricamento Excel
df = pd.read_excel(FILE_EXCEL)

print(f"Avvio elaborazione: {len(df)} righe trovate.")

with open(FILE_LABELS, mode='w', newline='') as f_label:
    writer = csv.writer(f_label)
    writer.writerow(['clip_name', 'fps', 'frame_inizio_crisi', 'frame_fine_crisi', 'frame_totali'])

    for index, row in df.iterrows():
        try:
            # Recupero nome video e aggiunta estensione se manca
            nome_video = str(row[COLONNA_FILE]).strip()
            if not nome_video.lower().endswith('.mp4'):
                nome_video += '.mp4'
            
            path_input = os.path.join(CARTELLA_VIDEO_INPUT, nome_video)
            
            ini_str = str(row[COLONNA_INIZIO]).strip()
            fine_str = str(row[COLONNA_FINE]).strip()
            
            t_inizio_crisi = hms_to_seconds(ini_str)
            t_fine_crisi = hms_to_seconds(fine_str)
            
            # Calcolo tempi di taglio con buffer
            start_taglio = max(0, t_inizio_crisi - BUFFER_SECONDI)
            durata_totale = (t_fine_crisi - t_inizio_crisi) + (BUFFER_SECONDI * 2)
            
            fps = get_video_info(path_input)
            output_name = f"clip_{index:03d}.mp4"
            path_output = os.path.join(CARTELLA_OUTPUT, output_name)

            # Comando FFmpeg (usa python3 per chiamarlo)
            comando = [
                'ffmpeg', '-y',
                '-ss', str(start_taglio),
                '-i', path_input,
                '-t', str(durata_totale),
                '-c:v', 'libx264', '-crf', '18', '-c:a', 'copy',
                path_output
            ]

            # Esecuzione silenziosa
            subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Calcolo frame per il labeling
            # Se lo start_taglio è > 0, l'inizio crisi è esattamente a BUFFER_SECONDI
            # Se start_taglio è 0 (perché la crisi era all'inizio), l'inizio è t_inizio_crisi
            offset_effettivo = t_inizio_crisi - start_taglio
            f_inizio = int(offset_effettivo * fps)
            f_fine = f_inizio + int((t_fine_crisi - t_inizio_crisi) * fps)
            f_totali = int(durata_totale * fps)

            writer.writerow([output_name, round(fps, 2), f_inizio, f_fine, f_totali])
            print(f"✅ Riga {index}: Creata {output_name} (Crisi: frame {f_inizio} -> {f_fine})")

        except Exception as e:
            print(f"Errore alla riga {index}: {e}")

print(f"\n Operazione completata! Dataset pronto in: {CARTELLA_OUTPUT}")