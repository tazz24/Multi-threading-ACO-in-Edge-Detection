from ImageProcessing.Process_Image import ProcessImage
from ACO.Ants_Colony import PartitionBasedAcs
import tkinter
from tkinter import *
from tkinter import filedialog as f_d
from tkinter import messagebox
from PIL import Image
from numpy import asarray

root = Tk()                                   # Liniile 10 - 15 crearea generala a meniului
root.title("Edge Detection Menu")
root.geometry('{}x{}'.format(600, 400))
root.resizable(None, None)#
root.config(background='#D3DEDC')
root.iconphoto(False, tkinter.PhotoImage(file='C:/Users/user/Desktop/LICENTA/Practic/educatie_icon.png'))


welcome_label = Label(root, text="Bine ati venit!", bg='#7C99AC', font=('Gothic', 20))
welcome_label.place(x=190, y=5)
frame_instructiuni = LabelFrame(root, text="Instructiuni", bg='#92A9BD', font=('Gothic', 20))
frame_instructiuni.place(x=10, y=40, height=150, width=350)

frame_diff = LabelFrame(root, text="Procentaj", bg='#92A9BD', font=('Gothic', 20))
frame_diff.place(x=10, y=200, height=150, width=350)

label_iteratii = Label(frame_instructiuni, text="Nr de iteratii:", bg='#7C99AC')
label_iteratii.config(font=('Arabic', 10))
label_iteratii.place(x=10, y=4)
my_box = Entry(frame_instructiuni)
my_box.place(x=150, y=4)

label_furnici = Label(frame_instructiuni, text="Nr de furnici:", bg='#7C99AC')
label_furnici.config(font=('Arabic', 10))
label_furnici.place(x=10, y=48)
my_box1 = Entry(frame_instructiuni)
my_box1.place(x=150, y=48)

label_furnici = Label(frame_instructiuni, text="Pasi de constructie:", bg='#7C99AC')
label_furnici.config(font=('Arabic', 10))
label_furnici.place(x=10, y=90)
my_box2 = Entry(frame_instructiuni)
my_box2.place(x=150, y=90)

label_imagini = Label(frame_diff, text="", bg='#92A9BD')
label_imagini.config(font=('Arabic', 10))
label_imagini.place(x=0, y=0)

label_procent = Label(frame_diff, text="", bg='#92A9BD')
label_procent.config(font=('Arabic', 10))
label_procent.place(x=0, y=20)


# Alegerea imaginii pentru comparare
def get_files():
    file_path = f_d.askopenfilename(initialdir='C:/Users/user/Desktop/LICENTA/Practic/RESULTS/IMG')
    return file_path


# Obtinrea procentului prin comparatia celor 2 imagini
def get_diff(cale):
    i1 = Image.open("C:/Users/user/Desktop/LICENTA/Practic/RESULTS/IMG/Initialization.png")
    pathhhh = ''.join(cale)
    i2 = Image.open(pathhhh)

    imagine1 = Image.fromarray(asarray(i1))
    imagine2 = Image.fromarray(asarray(i2))

    assert imagine1.mode == imagine2.mode, "Different kinds of images."
    assert imagine1.size == imagine2.size, "Different sizes."

    pairs = zip(imagine1.getdata(), imagine2.getdata())
    # Pentru imagini alb-negru
    dif = sum(abs(p1 - p2) for p1, p2 in pairs)
    ncomponents = imagine1.size[0] * imagine1.size[1] * 3
    rez = dif/255.0 * 100 / ncomponents
    print("Diferanta in procent: {}%".format((dif / 255.0 * 100) / ncomponents))

    label_procent.config(text=rez, bg='#7C99AC' )

# Afisarea suplimentara a informatiilor
def get_info():
    label_imagini.config(text="Pentru comparare -> apasati butonul din dreapta", bg='#7C99AC')


b1 = Button(frame_diff, text="Informatii", command=get_info)
b1.place(y=90, x=10)
b2 = Button(frame_diff, text="Obtinere procent", command=lambda: get_diff(get_files()))
b2.place(y=90, x=230)

# Clasa ' Detectarea muchiilor'
class EdgeDetection:
    def __init__(self, furnici, iteratii, pasi, cale):
        self.furnici = furnici
        self.iteratii = iteratii
        self.pasi = pasi
        self.cale = cale

    # Rularea aplicatiei
    def run(self):
        img_parse = ProcessImage(self.cale)
        heuristica = img_parse.ParsareIntensitati()
        ant = PartitionBasedAcs(self.furnici, 0.0001, 1, 0.1, 0.1, 0.05, 0.1, self.iteratii, self.pasi, 1, 1, heuristica, 8)
        ant.run()


# Introducerea nr. de iteratii de la meniu
def get_iteratii():
    user = my_box.get()
    return int(user)


# Introducerea nr. de furnici de la meniu
def get_furnici():
    user = my_box1.get()
    return int(user)


# Introducerea nr. de pasi de la meniu
def get_pasi():
    user = my_box2.get()
    return int(user)


#Alegerea imaginii de test ca data de intrare
def get_path():
    file_path = f_d.askopenfilename(initialdir='C:/Users/user/Desktop/LICENTA/Practic/ImageProcessing/Images')
    return file_path


# Pornirea aplicatiei
def get_start(furnici, iter, pasi, cale):
    edge = EdgeDetection(furnici, iter, pasi, cale)
    edge.run()

# Afisare informatii
def get_about():
    messagebox.showinfo("Despre", "Acesta a fost creat de Ciuclea Razvan")

# Afisare informatii
def get_instructiunui():
    messagebox.showinfo("Instructiunui", "Pentru incepere program,trebuie completata frameul de intstructiuni!!")

# Oprirea aplicatiei
def get_stop():
    root.destroy()


if __name__ == '__main__':
    # Meniul in care se va rula ACO prin multithreading
    frame_meniu = LabelFrame(root, text="Meniu", bg='#92A9BD', width=400, height=300, padx=50, pady=50, font=('Gothic', 20))
    frame_meniu.place(x=400, y=40)
    b1m = Button(frame_meniu, text="START", command=lambda: get_start(get_furnici(), get_iteratii(), get_pasi(), get_path()))  # pornire program
    b1m.pack(pady=10)
    b2m = Button(frame_meniu, text="DESPRE", command=get_about)  # detalii despre cine o facut
    b2m.pack(pady=10)
    b3m = Button(frame_meniu, text="INSTRUCTIUNI", command=get_instructiunui)  # legatura cu primul frame
    b3m.pack(pady=10)
    b4m = Button(frame_meniu, text='STOP', command=get_stop)  # oprire program
    b4m.pack(pady=10)

    root.mainloop()
