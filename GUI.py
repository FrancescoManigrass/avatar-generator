import PySimpleGUI as sg
import speech_recognition as sr
import cv2
import os
import subprocess


name = ""
height = ""
weight = ""
sex = ""
front_path = ""
side_path = ""

icon = 'D:/Tesi/Moro/icon_1.ico'

layout = [
    [sg.Text('Nome'), sg.InputText(size=(53, 1))],
    [sg.Text('Altezza (in m) '), sg.InputText(size=(46, 1))],
    [sg.Text('Peso (in kg)'), sg.InputText(size=(48, 1))],
    [sg.Text('Sesso'), sg.Radio('Maschio', 'Sesso'), sg.Radio('Femmina', 'Sesso')],
    [sg.Button('Scatta Foto')],
    [sg.Button('Genera Avatar', button_color=("Dark Blue", "white"), pad=((170, 0), (20, 0)))]
]

window = sg.Window('3DAvatarGenerator', icon=icon).Layout(layout)
#window.set_icon(icon)

# def take_photo():
#     # Codice per aprire la webcam e scattare una foto con comando vocale "scatta"
#     r = sr.Recognizer()
#     with sr.Microphone(device_index=1) as source:
#         sg.popup('Attendi il comando "scatta" per scattare una foto.')
#         audio = r.listen(source)
#     try:
#         text = r.recognize_google(audio, language='it-IT')
#         if text == 'scatta':
#             cap = cv2.VideoCapture(0)
#             ret, frame = cap.read()
#             if ret:
#                 cv2.imwrite('front.jpg', frame)

#                 #aggiungere ridimensionamento foto

#             cap.release()
#     except sr.UnknownValueError:
#         sg.popup('Comando vocale non riconosciuto. Riprova.')

# def take_photo():
#     # Codice per aprire la webcam e visualizzare l'anteprima
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow('Anteprima', frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

#         # Codice per scattare una foto con comando vocale "scatta"
#         r = sr.Recognizer()
#         #ha problemi perché pc senza microfono
#         with sr.Microphone(device_index=4) as source:
#             audio = r.listen(source)
#         try:
#             text = r.recognize_google(audio, language='it-IT')
#             if text == 'scatta':
#                 cv2.imwrite('front.jpg', frame)

#                 #aggiungere ridimensionamento foto

#                 sg.popup('Foto scattata con successo!')
#                 break
#         except sr.UnknownValueError:
#             pass

#     cap.release()
#     cv2.destroyAllWindows()


#versione per scattare con pulsante
def cattura_immagine():
    # Crea il layout della finestra principale
    layout = [
        [sg.Image(filename="", key="-WEBCAM-")],
        [sg.Button("Scatta", size=(10,1))]
    ]

    # Crea la finestra principale
    window = sg.Window("Webcam", layout, location=(450, 100), icon=icon)
    
    # Crea il loop dell'applicazione
    while True:
        event, values = window.read(timeout=20)

        # Se l'utente chiude la finestra, esci dal loop
        if event == sg.WIN_CLOSED:
            break

        # Cattura l'immagine dalla webcam e mostra l'anteprima nell'elemento Image
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-WEBCAM-"].update(data=imgbytes)
        #cap.release()

        counter=0

        # Se l'utente preme il pulsante "Scatta", cattura l'immagine e salvala come file
        if event == "Scatta":
            # Apri la webcam
            #cap = cv2.VideoCapture(0)
            # Cattura un fotogramma dalla webcam
            #ret, frame = cap.read()

            if int(counter)==0:
                # Salva l'immagine come file "foto.png"
                cv2.imwrite(front_path, frame)
                #aggiungere ridimensionamento foto
                #aggiustare percorsi
                command = f"python resize.py --path {front_path}"
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            else:
                cv2.imwrite(side_path, frame)
                command = f"python resize.py --path {side_path}"
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

            counter=counter+1

            if int(counter)==2:
                # Chiudi la finestra della webcam
                cap.release()
                window.close()

    # Chiudi la finestra principale
    cap.release()
    window.close()




def validate_data(values):
    # Validazione dei dati inseriti dall'utente
    if not values[0]:
        sg.popup('Inserisci il nome', icon=icon)
        return False
    if not values[1].isdigit() or not values[2].isdigit():
        sg.popup('Inserisci un valore numerico per altezza e peso', icon=icon)
        return False
    if not values[3]:
        sg.popup('Seleziona il sesso', icon=icon)
        return False
    return True




def main():


    while True:
        event, values = window.Read()

        if event == sg.WIN_CLOSED:
            break

        elif event == 'Scatta Foto':
            cattura_immagine()
            sg.popup('Foto scattata con successo!', icon=icon)

        elif event == 'Genera Avatar':
            # Recupera i valori inseriti dall'utente
            name = values['-NAME-']
            height = values['-HEIGHT-']
            weight = values['-WEIGHT-']
            gender = values['-SEX-']

            if validate_data(values):
                # Elaborazione dei dati inseriti e lancio dello script Python

                #aggiungere chiamata allo script demo.py
                command = f"python demo.py --gender {gender} --height {height} --weight {weight} --front {front_path} --side {side_path} --mesh_name {name} --experiment {name}"
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                sg.popup(f'Complimenti {name}, il tuo avatar è stato creato!', icon=icon)


if __name__ == '__main__':
    main()