from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

t0 = time.time()

app = Tk()
app.geometry("2560x1600+0+0")
app.title("RöntgeNet Software")

frontal_image_path = ""
lateral_image_path = ""

frontal_img = ImageTk.PhotoImage(Image.open("./im/kf.png"))
frontal_panel = Label(app, image=frontal_img)
frontal_panel.place(x = 20, y = 40)

def frontal_image(): #load frontal image / get path
    path = filedialog.askopenfilename(filetypes=[("Image File", '.jpg')])
    global frontal_image_path
    frontal_image_path = path #save file for calculation

    img2 = ImageTk.PhotoImage(Image.open(path))
    frontal_panel.configure(image=img2)
    frontal_panel.image = img2

frontal = Button(app, text="Frontale Aufnahme wählen", command=frontal_image)
frontal.place(x = 90, y = 380)

lateral_img = ImageTk.PhotoImage(Image.open("./im/kf.png"))
lateral_panel = Label(app, image=frontal_img)
lateral_panel.place(x = 380, y = 40)

def lateral_image(): #load frontal image / get path
    path = filedialog.askopenfilename(filetypes=[("Image File", '.jpg')])
    global lateral_image_path
    lateral_image_path = path #save file for calculation

    img2 = ImageTk.PhotoImage(Image.open(path))
    lateral_panel.configure(image=img2)
    lateral_panel.image = img2

seitlich = Button(app, text="Seitliche Aufnahme wählen", command=lateral_image)
seitlich.place(x = 450, y = 380)

predict_img = ImageTk.PhotoImage(Image.open("./im/kp.png").resize((640, 480)))
predict_panel = Label(app, image=predict_img)
predict_panel.place(x = 740, y = 40)


DN121 = tf.keras.models.load_model('./f_models/CXRNet1.1-f-DN121_2')
l_DN121 = tf.keras.models.load_model('./f_models/CXRNet1.1-l-DN121_2')

atel_l = Label(app, text="0.00 - Atelektase")
atel_l.place(x = 20, y = 550)
kard_l = Label(app, text="0.00 - Kardiomegalie")
kard_l.place(x = 20, y = 570)
kons_l = Label(app, text="0.00 - Konsolidierung")
kons_l.place(x = 20, y = 590)
odem_l = Label(app, text="0.00 - Ödem")
odem_l.place(x = 20, y = 610)
vkar_l = Label(app, text="0.00 - Vergrößertes Kardiomastinum")
vkar_l.place(x = 20, y = 630)
frak_l = Label(app, text="0.00 - Fraktur")
frak_l.place(x = 20, y = 650)
lasi_l = Label(app, text="0.00 - Lungenläsion")
lasi_l.place(x = 20, y = 670)
trub_l = Label(app, text="0.00 - Lungentrübung")
trub_l.place(x = 20, y = 690)
kein_l = Label(app, text="0.00 - Kein Befund")
kein_l.place(x = 20, y = 710)
pleu_l = Label(app, text="0.00 - Pleuraerguss")
pleu_l.place(x = 20, y = 730)
ande_l = Label(app, text="0.00 - Anderer Pleuraler Befund")
ande_l.place(x = 20, y = 750)
entz_l = Label(app, text="0.00 - Lungenentzündung")
entz_l.place(x = 20, y = 770)
pneu_l = Label(app, text="0.00 - Pneumothorax")
pneu_l.place(x = 20, y = 790)
stut_l = Label(app, text="0.00 - Stützvorrichtungen")
stut_l.place(x = 20, y = 810)


def loadTable(): #process frontal + lateral image, calculate result.

    if frontal_image_path == "":
        messagebox.showinfo('Frontalaufnahme Fehlt', 'Die Frontalaufnahme muss noch bestimmt werden.')
        return

    t0 = time.time()

    f_train_images = []

    # load image
    f_im = Image.open(frontal_image_path)
    f_matrix = np.array(f_im.getdata())
    f_normalized_matrix = np.true_divide(f_matrix, 255)
    f_zero_mean_mtrx = f_normalized_matrix - np.mean(f_normalized_matrix)
    f_standardized_matrix = f_zero_mean_mtrx / np.std(f_normalized_matrix)

    f_train_images.extend(f_standardized_matrix)

    frontal_prepared = np.asarray(f_train_images).reshape(1, 320, 320, 1)

    result = DN121.predict(frontal_prepared)[0]

    if lateral_image_path != "":
        l_train_images = []

        l_im = Image.open(lateral_image_path)
        l_matrix = np.array(l_im.getdata())
        l_normalized_matrix = np.true_divide(l_matrix, 255)
        l_zero_mean_mtrx = l_normalized_matrix - np.mean(l_normalized_matrix)
        l_standardized_matrix = l_zero_mean_mtrx / np.std(l_normalized_matrix)

        l_train_images.extend(l_standardized_matrix)
        lateral_prepared = np.asarray(l_train_images).reshape(1, 320, 320, 1)
        l_result = l_DN121.predict(lateral_prepared)[0]
        result = [(result[i] + l_result[i])/2 for i in range(14)]

    class_names = ['Atelektase', 'Kardiomegalie', 'Konsolidierung', 'Ödem', 'Vergrößertes Kardiomastinum', 'Fraktur',
                   'Lungenläsion', 'Lungentrübung', 'Kein Befund', 'Pleuraerguss', 'Anderer Pleuraler Befund', 'Lungenentzündung',
                   'Pneumothorax', 'Stützvorrichtungen']

    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(left=0.4)

    # Example data
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, result, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Ausgang')
    ax.set_ylabel('Befund')
    ax.set_title('RöntgeNet Prognose')

    plt.savefig("./im/predict.png")

    img2 = ImageTk.PhotoImage(Image.open("./im/predict.png"))
    predict_panel.configure(image=img2)
    predict_panel.image = img2

    t1 = time.time()
    total = t1-t0

    atel_l.config(text="{} - Atelektase".format(int(result[0]*100)/100))
    kard_l.config(text="{} - Kardiomegalie".format(int(result[1]*100)/100))
    kons_l.config(text="{} - Konsolidierung".format(int(result[2]*100)/100))
    odem_l.config(text="{} - Ödem".format(int(result[3]*100)/100))
    vkar_l.config(text="{} - Vergrößertes Kardiomastinum".format(int(result[4]*100)/100))
    frak_l.config(text="{} - Fraktur".format(int(result[5]*100)/100))
    lasi_l.config(text="{} - Lungenläsion".format(int(result[6]*100)/100))
    trub_l.config(text="{} - Lungentrübung".format(int(result[7]*100)/100))
    kein_l.config(text="{} - Kein Befund".format(int(result[8]*100)/100))
    pleu_l.config(text="{} - Pleuraerguss".format(int(result[9]*100)/100))
    ande_l.config(text="{} - Anderer Pleuraler Befund".format(int(result[10]*100)/100))
    entz_l.config(text="{} - Lungenentzündung".format(int(result[11]*100)/100))
    pneu_l.config(text="{} - Pneumothorax".format(int(result[12]*100)/100))
    stut_l.config(text="{} - Stützvorrichtungen".format(int(result[13]*100)/100))

    calc_label.config(text="{} - Berechnungszeit".format(int(total*1000)))

berechnen = Button(app, text="Berechnen", command=loadTable)
berechnen.place(x = 1040, y = 540)

def reset():
    global frontal_image_path
    frontal_image_path = ""

    global lateral_image_path
    lateral_image_path = ""

    img2 = ImageTk.PhotoImage(Image.open("./im/kf.png"))
    frontal_panel.configure(image=img2)
    frontal_panel.image = img2

    lateral_panel.configure(image=img2)
    lateral_panel.image = img2

    img3 = ImageTk.PhotoImage(Image.open("./im/kp.png"))
    predict_panel.configure(image=img3)
    predict_panel.image = img3

    atel_l.config(text="0.00 - Atelektase")
    kard_l.config(text="0.00 - Kardiomegalie")
    kons_l.config(text="0.00 - Konsolidierung")
    odem_l.config(text="0.00 - Ödem")
    vkar_l.config(text="0.00 - Vergrößertes Kardiomastinum")
    frak_l.config(text="0.00 - Fraktur")
    lasi_l.config(text="0.00 - Lungenläsion")
    trub_l.config(text="0.00 - Lungentrübung")
    kein_l.config(text="0.00 - Kein Befund")
    pleu_l.config(text="0.00 - Pleuraerguss")
    ande_l.config(text="0.00 - Anderer Pleuraler Befund")
    entz_l.config(text="0.00 - Lungenentzündung")
    pneu_l.config(text="0.00 - Pneumothorax")
    stut_l.config(text="0.00 - Stützvorrichtungen")

    calc_label = Label(app, text="0 - Berechnungszeit")


entfernen = Button(app, text="Zurücksetzen", command=reset)
entfernen.place(x = 1280, y = 10)

gh_link = Label(app, text="RöntgeNet Software", font='TkDefaultFont 14 bold')
gh_link.place(x = 20, y = 10)

gh_lin = Label(app, text="github.com/zanderfilet/R-ntgeNet_Software", fg="blue")
gh_lin.place(x = 600, y = 10)

time_label = Label(app, text="Geschwindigkeit (ms)", font='TkDefaultFont 14 bold')
time_label.place(x = 20, y = 420)

t1 = time.time()
total = t1-t0

config_label = Label(app, text="{} - Konfigurationszeit".format(int(total*1000)))
config_label.place(x = 20, y = 450)
calc_label = Label(app, text="0 - Berechnungszeit")
calc_label.place(x = 20, y = 480)

#########

header_l = Label(app, text="Quantitative Resultate", font='TkDefaultFont 14 bold')
header_l.place(x = 20, y = 525)

mainloop()