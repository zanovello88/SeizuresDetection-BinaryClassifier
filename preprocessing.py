#Codice per realizzare il pre-processing.
#Questo codice prende il video specificato in video_input e taglia il video realizzando un video 210x210 pixel
#del singolo topo, quindi da un video unico con 8 topi se ne generano 8 che vengono salvati nelle relativi cartelle.
#Il taglio avviene per tutta la durata del video. 
import os
import subprocess

#1. CONFIGURAZIONE MANUALE
video_input = "05-03-2025 20-2030 2025-11-12 17-14-16.mp4" 
output_base_dir = "dataset_tesi_topi"

#2. PARAMETRI FISSI (NON CAMBIARE)
W, H = 210, 210
coordinate_gabbie = [
    (80, 120), (265, 120), (570, 120), (780, 120),  # Riga sopra
    #(80, 325), (265, 325), (570, 325), (780, 325)   # Riga sotto
    (570, 325), (780, 325) #Se manca A5 e A3
]
nomi_topi = ["78", "83", "A1", "AL", "P95", "A7"] #Se manca A5 e A3
#nomi_topi = ["78", "83", "A1", "AL", "A5", "A3", "P95", "A7"]

#3. LOGICA DI ELABORAZIONE
def split_video_manual():
    if not os.path.exists(video_input):
        print(f"Errore: Il file '{video_input}' non è stato trovato nella cartella.")
        return

    # Estraiamo il nome del video (senza .mp4) per differenziare i file nelle cartelle
    video_label = os.path.splitext(video_input)[0]
    
    print(f"Inizio elaborazione integrale del video: {video_input}")

    for nome, (x, y) in zip(nomi_topi, coordinate_gabbie):
        # Cartella specifica del topo (es: dataset_tesi_topi/topo_78)
        topo_dir = os.path.join(output_base_dir, f"topo_{nome}")
        if not os.path.exists(topo_dir):
            os.makedirs(topo_dir)
            
        # Nome file finale (es: topo_78/78_video_giorno_1.mp4)
        output_file = os.path.join(topo_dir, f"{nome}_{video_label}.mp4")
        
        print(f"📹 Taglio Topo {nome} (x={x}, y={y})...")

        command = [
            'ffmpeg',
            '-i', video_input,
            #'-t', '60', vincolo di tempo per fare test 
            '-vf', f'crop={W}:{H}:{x}:{y}',
            '-c:v', 'libx264',
            '-crf', '18',      # Qualità visiva ottima per analisi IA
            '-preset', 'fast', 
            '-c:a', 'copy',    # Copia l'audio se presente (non rallenta)
            output_file,
            '-y'
        ]
        
        try:
            # Esecuzione
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"✅ Salvato: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Errore durante il processing del topo {nome}: {e}")

    print(f"\n Elaborazione di '{video_input}' completata con successo!")

if __name__ == "__main__":
    split_video_manual()