# Import the required Libraries
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk, ImageFilter

import Code
import cv2

import PIL.Image
import PIL.ImageTk

# import ShowImage

file_name = "null"
option = "null" 


def show(file_name, title="Input Image"):
    print("started show() \t", file_name)

    # Create a window
    window = tk.Toplevel()
    window.title(title)

    img = cv2.imread(file_name)
    cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
    height, width, no_channels = cv_img.shape
    # Create a canvas that can fit the above image
    canvas = tk.Canvas(window, width=width, height=height)
    canvas.pack()
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
    # Add a PhotoImage to the Canvas
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    window.mainloop()


def open_file():
    print("open_file fun started\n")

    global file_name

    file = filedialog.askopenfilename()

    file_name = file

    show(file_name)


def enhanceImage():
    global file_name, option

    print("started enhanceImage() \n")
    print(option, "\n")

    if(Code.isGreyScale(file_name)):
        new_file_name = Code.enhanceGreyscaleImage(file_name)
        show(new_file_name)

        return
    if(option == "null"):
        print("Please select one option\n")
        return
    elif(option == "traditional"):
        print("if option == traditional\n")
        new_file_name = Code.traditionalMethod(file_name)
        show(new_file_name)
        return

    new_file_name = Code.enhanceColorImage(file_name)

    show(new_file_name)

    print("stopped enhanceImage() \n")


# functions for selecting options
def selectTraditional():
    print("traditional selected")

    global option
    option = "traditional"


def selectAdaptive():
    print("adaptive selected")

    global option
    option = "adaptive"


def selectCLAHE():
    print("clahe selected")

    global option
    option = "clahe"


def main():
    # Create an instance of tkinter frame
    root = tk.Tk()

    # Give title to the tkinter frame
    root.title('Image Enhancement')

    # Create canvas
    canvas = tk.Canvas(root, bg='#DDA0DD', height=10000, width=10000)
    canvas.pack()

    # background_image = tk.PhotoImage('Black')
    # background_label = tk.Label(root, image=background_image)
    # background_label.place(relwidth=1, relheight=1)

    # Create different frames into the canvas
    # Our model
    main_frame = tk.Frame(root, bg='#595959', bd=10)
    # Set the position of frame
    main_frame.place(relx=0.5, rely=0, relwidth=0.90,
                     relheight=0.1, anchor='n')
    # Give label for the frame
    model = tk.Label(main_frame, text="Our Model",
                     bg="white", font=('calibri', 20, 'bold'))
    # Set geometry of the label
    model.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.6)

    frame = tk.Frame(root, bg='#DDA0DD', bd=10)
    frame.place(relx=0.5, rely=0.2, relwidth=0.90, relheight=0.1, anchor='n')

    frame1 = tk.Frame(root, bg='#595959', bd=10)
    frame1.place(relx=0.5, rely=0.1, relwidth=0.90, relheight=0.1, anchor='n')

    # frame_a = tk.Frame(root, bg='#595959', bd=10)
    # frame_a.place(relx=0.08, rely=0.1, relwidth=0.1, relheight=0.7, anchor='n')

    # Definition of the open_file function

    # vertical

    # blur = tk.Button(frame, text='Blur', command=lambda: Blur(path), font=('calibri', 20, 'bold'))
    # blur.place(relx=0.4, relheight=0.9, relwidth=0.2)

    # Menu bar

    menu2 = tk.Button(frame1, text='Traditional', command=selectTraditional,
                      bg="white", font=('calibri', 20, 'bold'))
    menu2.place(relx=0.3, rely=0, relheight=0.7, relwidth=0.2)

    menu3 = tk.Button(frame1, text='Adaptive', command=selectAdaptive,
                      bg="white", font=('calibri', 20, 'bold'))
    menu3.place(relx=0.5, rely=0, relheight=0.7, relwidth=0.2)

    menu4 = tk.Button(frame1, text='CLAHE', command=selectCLAHE,
                      bg="white", font=('calibri', 20, 'bold'))
    menu4.place(relx=0.7, rely=0, relheight=0.7, relwidth=0.2)

    # Create button to select the input image
    input_img = tk.Button(frame, text='Browse', command=open_file,
                          bg="yellow", font=('calibri', 20, 'bold'))
    input_img.place(relx=0.2, rely=0.2, relheight=0.7, relwidth=0.2)

    # output_img = tk.Button(frame, text='Greyscale', command=open_file,bg="grey", font=('calibri', 20, 'bold'))
    # output_img.place(relx=0.4, relheight=0.9, relwidth=0.2)

    # Create button to equalize the inputed image
    output_img1 = tk.Button(frame, text='Enhance', command=lambda: enhanceImage(
    ), bg="yellow", font=('calibri', 20, 'bold'))

    output_img1.place(relx=0.6, rely=0.2, relheight=0.7, relwidth=0.2)

    # Create lower frame for input image
    lower_frame = tk.Frame(root, bg='#595959', bd=10)
    # Set geometry for the frame
    lower_frame.place(relx=0.3, rely=0.3, relwidth=0.30,
                      relheight=0.6, anchor='n')
    bg_color = 'white'

    # Create lower frame for enhanced image
    lower_frame1 = tk.Frame(root, bg='#595959', bd=10)
    # Set Geometry of the frame
    lower_frame1.place(relx=0.7, rely=0.3, relwidth=0.30,
                       relheight=0.6, anchor='n')

    bg_color1 = 'white'

    # Create canvas to display the enhanced image
    output1 = tk.Canvas(lower_frame1, bg=bg_color1,
                        bd=10, highlightthickness=0)
    # Set the geometry of canvas
    output1.place(relx=0, rely=0, relwidth=1, relheight=0.9)
    # Give label to the output canvas
    output_label = tk.Label(lower_frame1, text="Output image",
                            bg="yellow", font=('calibri', 20, 'bold'))
    output_label.place(relx=0, rely=0.9, relwidth=1, relheight=0.1)

    # Start the tkinter frame
    root.mainloop()


main()
