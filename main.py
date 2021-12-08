# Import the required Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import Code
import time

# Create an instance of tkinter frame
root = tk.Tk()

file_name = "null"
option = "null"


def show(file_name, title="Input Image"):
    global root

    print("started show() \t", file_name)

    # Create a window
    window = tk.Toplevel()
    window.title(title)

    img = cv2.imread(file_name)

    cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))

    # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
    height, width, no_channels = cv_img.shape

    frame = tk.Frame(window, bg='#595959', bd=1,
                     width=width*2, height=height*2)
    frame.pack()

    # frame.place(relx=0, rely=0, relwidth=1,
    #               relheight=1, anchor='n')

    # Create a canvas that can fit the above image
    canvas = tk.Canvas(frame, width=width, height=height)
    canvas.pack()

    # Add a PhotoImage to the Canvas

    window.photo = photo

    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# function to show the text in a canvas


def showText(canvas, text):
    canvas.delete('all')
    canvas.create_text(
        200, 100,
        fill="darkblue",
        font="Times 20 italic bold",
        text=text)
    canvas.pack()


# function to get parameters of an image such as noise and blurryness

def getParametersOfAnImage(file_name):
    image = cv2.imread(file_name)
    noise = Code.getNoise(image)
    blurryness = Code.getBlurryness(image)

    return noise, blurryness


def open_file():
    print("open_file fun started\n")

    global file_name

    file = filedialog.askopenfilename()

    if(file == ""):
        print("Please select an image...")
        # Show message in the window

        return

    file_name = file

    show(file_name)


def enhanceImage(canvas_input, canvas_output):
    global file_name, option

    print("started enhanceImage() \n")
    print(option, "\n")

    if(option == "null"):
        print("Please select one option\n")
        showText(canvas_input, "Please select one option")
        return

    elif(option == "traditional"):

        print("if option == traditional\n")

        if(Code.isGreyScale(file_name)):
            new_file_name = Code.traditionalMethodForGreyScale(file_name)

        else:
            new_file_name = Code.traditionalMethod(file_name)

        # get histogram of old image
        histogram_of_old_image = Code.getHistogram(file_name)

        # get histogram of new processed image
        histogram_of_new_image = Code.getHistogram(new_file_name)

        show(new_file_name, "Traditional")

        show(histogram_of_old_image, "Histogram " + file_name)
        show(histogram_of_new_image, "Histogram " + new_file_name)

        noise_input_image, blurryness_input_image = getParametersOfAnImage(
            file_name)
        noise_output_image, blurryness_output_image = getParametersOfAnImage(
            new_file_name)

        showText(canvas_input, "Noise : " + str(noise_input_image) +
                 "\n" + "Blurryness " + str(blurryness_input_image))
        showText(canvas_output, "Noise : " + str(noise_output_image) +
                 "\n" + "Blurryness " + str(blurryness_output_image))

    elif(option == "clahe"):

        if(Code.isGreyScale(file_name)):
            print("Image is grey scale\n")
            # apply CLAHE for grey scale
            new_file_name = Code.claheGreyscaleMethod(file_name)

        else:
            # apply CLAHE algorithm
            new_file_name = Code.CLAHE(file_name)

        # get histogram of old image
        histogram_of_old_image = Code.getHistogram(file_name)

        # get histogram of new processed image
        histogram_of_new_image = Code.getHistogram(new_file_name)

        show(new_file_name, "Adaptive Method")

        show(histogram_of_old_image, "Histogram " + file_name)
        show(histogram_of_new_image, "Histogram " + new_file_name)

        noise_input_image, blurryness_input_image = getParametersOfAnImage(
            file_name)
        noise_output_image, blurryness_output_image = getParametersOfAnImage(
            new_file_name)

        showText(canvas_input, "Noise : " + str(noise_input_image) +
                 "\n" + "Blurryness " + str(blurryness_input_image))
        showText(canvas_output, "Noise : " + str(noise_output_image) +
                 "\n" + "Blurryness " + str(blurryness_output_image))

    elif(option == "ourmethod"):

        if(Code.isGreyScale(file_name)):
            # apply our method for grey scale
            new_file_name = Code.enhanceGreyscaleMethod(file_name)

        else:
            # apply our method algorithm
            new_file_name = Code.ourMethod(file_name)

        # get histogram of old image
        histogram_of_old_image = Code.getHistogram(file_name)

        # get histogram of new processed image
        histogram_of_new_image = Code.getHistogram(new_file_name)

        show(new_file_name, "Our Method")

        show(histogram_of_old_image, "Histogram " + file_name)
        show(histogram_of_new_image, "Histogram " + new_file_name)

        noise_input_image, blurryness_input_image = getParametersOfAnImage(
            file_name)
        noise_output_image, blurryness_output_image = getParametersOfAnImage(
            new_file_name)

        showText(canvas_input, "Noise : " + str(noise_input_image) +
                 "\n" + "Blurryness " + str(blurryness_input_image))
        showText(canvas_output, "Noise : " + str(noise_output_image) +
                 "\n" + "Blurryness " + str(blurryness_output_image))

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


def selectOurs():
    print("ours selected")

    global option
    option = "ourmethod"


def main():
    global root
    # Give title to the tkinter frame
    root.title('Image Enhancement')

    # Create canvas
    canvas = tk.Canvas(root, bg='#DDA0DD', height=10000, width=10000)
    canvas.pack()

    # Create different frames into the canvas
    # Our model
    main_frame = tk.Frame(root, bg='#595959', bd=10)
    # Set the position of frame
    main_frame.place(relx=0.5, rely=0, relwidth=0.90,
                     relheight=0.1, anchor='n')
    # Give label for the frame
    main_frame_label = tk.Label(main_frame, text="Our Model",
                                bg="white", font=('calibri', 20, 'bold'))

    # Set geometry of the label
    main_frame_label.place(relx=0.4, rely=0, relwidth=0.2, relheight=0.6)

    frame = tk.Frame(root, bg='#DDA0DD', bd=10)
    frame.place(relx=0.5, rely=0.2, relwidth=0.90, relheight=0.1, anchor='n')

    frame1 = tk.Frame(root, bg='#595959', bd=10)
    frame1.place(relx=0.5, rely=0.1, relwidth=0.90, relheight=0.1, anchor='n')

    # Menu bar

    button_traditional = tk.Button(frame1, text='Traditional', command=selectTraditional,
                                   bg="white", font=('calibri', 20, 'bold'))
    button_traditional.place(relx=0.3, rely=0, relheight=0.7, relwidth=0.2)

    button_adaptive = tk.Button(frame1, text='Adaptive', command=selectCLAHE,
                                bg="white", font=('calibri', 20, 'bold'))
    button_adaptive.place(relx=0.5, rely=0, relheight=0.7, relwidth=0.2)

    button_clahe = tk.Button(frame1, text='OURS', command=selectOurs,
                             bg="white", font=('calibri', 20, 'bold'))
    button_clahe.place(relx=0.7, rely=0, relheight=0.7, relwidth=0.2)

    # Create button to select the input image
    button_input_image = tk.Button(frame, text='Browse', command=open_file,
                                   bg="yellow", font=('calibri', 20, 'bold'))
    button_input_image.place(relx=0.2, rely=0.2, relheight=0.7, relwidth=0.2)

    bg_color = 'white'
    bg_color1 = 'white'

    # Create lower frame for input image parameters
    lower_frame_for_input = tk.Frame(root, bg='#595959', bd=10)

    # Set geometry for the frame
    lower_frame_for_input.place(relx=0.3, rely=0.3, relwidth=0.30,
                                relheight=0.6, anchor='n')

    # Create lower frame for enhanced image parameters
    lower_frame_for_output = tk.Frame(root, bg='#595959', bd=10)

    # Set Geometry of the frame
    lower_frame_for_output.place(relx=0.7, rely=0.3, relwidth=0.30,
                                 relheight=0.6, anchor='n')

    # Create canvas to display the input image
    canvas_input = tk.Canvas(lower_frame_for_input, bg=bg_color,
                             bd=10, highlightthickness=0)
    canvas_input.place(relx=0, rely=0, relwidth=1, relheight=0.9)

    # Give label to the output canvas
    input_label = tk.Label(lower_frame_for_input, text="Input Image Parameters",
                           bg="yellow", font=('calibri', 20, 'bold'))

    input_label.place(relx=0, rely=0.9, relwidth=1, relheight=0.1)

    # Create canvas to display the enhanced image
    canvas_output = tk.Canvas(lower_frame_for_output, bg=bg_color1,
                              bd=10, highlightthickness=0)

    # Set the geometry of canvas
    canvas_output.place(relx=0, rely=0, relwidth=1, relheight=0.9)

    # Create button to equalize the inputed image
    button_enhance_image = tk.Button(frame, text='Enhance', command=lambda: enhanceImage(canvas_input, canvas_output
                                                                                         ), bg="yellow", font=('calibri', 20, 'bold'))

    button_enhance_image.place(relx=0.6, rely=0.2, relheight=0.7, relwidth=0.2)

    # Give label to the output canvas
    output_label = tk.Label(lower_frame_for_output, text="Output Image Parameters",
                            bg="yellow", font=('calibri', 20, 'bold'))
    output_label.place(relx=0, rely=0.9, relwidth=1, relheight=0.1)

    # Start the tkinter frame
    root.mainloop()


main()


# CHANGE NAMING OF FRAMES
# HISTOGRAM
# histogram overlay
# change saved images name prefix for every method
