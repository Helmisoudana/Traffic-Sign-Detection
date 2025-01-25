
import tkinter as tk
from tkinter import messagebox
from tkinter import Canvas
import threading
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import time
import os  # For sound notifications
from keras.models import load_model
# Global variable for spinner animation
detection_running = False
output_folder = 'detected_signs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
threshold = 0.90  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model =load_model('traffif_sign_model.h5' ,compile=False)
# Function to simulate a loading spinner (text-based)
def animate_loading_spinner():
    spinner_symbols = ["⏳", "⌛", "⏲️", "⏱️"]
    idx = 0

    while detection_running:
        loading_spinner.config(text=spinner_symbols[idx % len(spinner_symbols)])
        idx += 1
        time.sleep(0.5)
        loading_spinner.update()

def start_loading_spinner():
    global detection_running
    detection_running = True
    threading.Thread(target=animate_loading_spinner).start()

def stop_loading_spinner():
    global detection_running
    detection_running = False
# Function to exit the application
def quit_app():
    global detection_running
    detection_running = False
    root.quit()

# Main window

root = tk.Tk()
root.title("Traffic Sign Detection")
root.geometry("800x600")

# Load and display the background image
bg_image = Image.open(r"name.png")
bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(root, width=800, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title Label
title_label = tk.Label(
    root, text="Traffic Sign Detection", font=("Helvetica", 26, "bold"), fg="#FFD700", bg="#0F3460"
)
title_label.place(relx=0.5, y=40, anchor="center")
# Result Display (Label for Sign Image - Transparent background)
result_label = tk.Label(
    root, bg=root.cget("background")  # Set background to transparent
)
result_label.place(relx=0.5, rely=0.6, anchor="center")
# Icon Frame (Placeholder for Sign Icons - Transparent Label)
icon_frame = tk.Frame(root, bg="#0F3460")
icon_frame.place(relx=0.5, rely=0.8, anchor="center")

# Loading Spinner (Detection symbol)
loading_label = tk.Label(
    root, text="Detecting...", font=("Helvetica", 14, "italic"), fg="#ECF0F1", bg="#0F3460"
)
loading_spinner = tk.Label(
    root, text="⏳", font=("Helvetica", 24), fg="#FFD700", bg="#0F3460"
)
loading_label.place(relx=0.5, rely=0.9, anchor="center")
loading_spinner.place(relx=0.5, rely=0.95, anchor="center")

# Remove all buttons
# No buttons now, as per your request

image_folder_path = "gtsrb-german-traffic-sign/Meta"
output_folder = 'detected_signs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
threshold = 0.90  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = load_model('traffif_sign_model.h5' ,compile=False)

# Charger toutes les images dans une liste
images_list = []

def load_images():
    global images_list
    # Charger les images numérotées de 0.png à 42.png dans l'ordre
    for i in range(43):  # 43 images, de 0.png à 42.png
        image_path = os.path.join(image_folder_path, f"{i}.png")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            images_list.append(img)
        else:
            images_list.append(None)  # Pour éviter les erreurs si une image est manquante

# Fonction pour afficher une image en fonction de l'indice
def display_image_by_index(index):
    if 0 <= index < len(images_list) and images_list[index] is not None:
        result_label.config(image=images_list[index])
        result_label.image = images_list[index]
    else:
        result_label.config(text="Image non disponible pour cet indice.")

# Charger les images une fois
load_images()

def preprocess_img(imgBGR, erode_dilate=True):  # pre-processing fro detect signs in  image.
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin

#Counting the number of signs in the image
def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

#preprocessing the image before feeding it to the model
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

#Defining the labels
def getCalssName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'
    

def process_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Réduire la résolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Réduire la résolution
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, img = cap.read()
        frame_count += 1
        current_time = time.time()

        # Capture une image toutes les 2 secondes
        if current_time - start_time >= 2:
            start_time = current_time

            # Traitement de l'image pour détecter les panneaux
            img_bin = preprocess_img(img, False)
            min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
            rects = contour_detect(img_bin, min_area=min_area)
            img_bbx = img.copy()

            for rect in rects:
                xc = int(rect[0] + rect[2] / 2)
                yc = int(rect[1] + rect[3] / 2)
                size = max(rect[2], rect[3])
                x1 = max(0, int(xc - size / 2))
                y1 = max(0, int(yc - size / 2))
                x2 = min(cols, int(xc + size / 2))
                y2 = min(rows, int(yc + size / 2))

                if rect[2] > 100 and rect[3] > 100:
                    cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                crop_img = np.asarray(img[y1:y2, x1:x2])
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = preprocessing(crop_img)
                crop_img = crop_img.reshape(1, 32, 32, 1)
                predictions = model.predict(crop_img)
                classIndex = np.argmax(predictions, axis=-1)
                probabilityValue = np.amax(predictions)

                if probabilityValue > threshold:
                    # Ajout des labels sur l'image
                    cv2.putText(img_bbx, str(classIndex) + " " + str(getCalssName(classIndex)), 
                                (rect[0], rect[1] - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", 
                                (rect[0], rect[1] - 40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Sauvegarde de l'image dans le dossier
                    image_name = os.path.join(output_folder, f"detected_sign_{frame_count}.png")
                    cv2.imwrite(image_name, img_bbx)
                    print(f"Image sauvegardée: {image_name} ",int(classIndex))
                    display_image_by_index(int(classIndex))
                    break  # Arrêter après avoir trouvé un panneau pour cette image

            cv2.imshow("Output", img_bbx)  # Afficher le résultat

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Arrêter le programme en appuyant sur 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
video_thread = threading.Thread(target=process_video)
video_thread.daemon = True  # Le thread s'arrête lorsque la fenêtre Tkinter est fermée
video_thread.start()



root.mainloop()
