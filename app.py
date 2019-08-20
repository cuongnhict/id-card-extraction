import warnings
import os
from id_card_reader import IdCardReader
from PIL import Image as PILImage
from PIL.ImageTk import PhotoImage as PILPhotoImage
from tkinter import *
from tkinter import filedialog

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Application(Frame):
    def __init__(self, root):
        # Initial variables
        self.root = root
        self.image = None
        self.file_name = None
        self.id_reader = IdCardReader()

        Frame.__init__(self, self.root, background='white')

        self.frame_left = Frame(self)
        self.frame_left.rowconfigure(1, pad=15)
        self.frame_left.grid(row=0, column=0, sticky=(N, W, E, S))

        self.canvas = Canvas(self.frame_left, highlightthickness=1, highlightbackground='#ccc')
        self.canvas.bind("<ButtonRelease-1>", self.open_image)
        self.canvas.config(width=400, height=400)
        self.canvas.grid(row=0, column=0, padx=(10, 0), pady=10)

        self.frame_right = Frame(self, padx=10, pady=10)
        self.frame_right.rowconfigure(1, pad=15)
        self.frame_right.grid(row=0, column=1, sticky=(N, W, E, S))

        self.lbl1 = Label(self.frame_right, text='ID:')
        self.lbl1.grid(row=0, column=0, sticky=W)

        self.id_number = StringVar()
        self.lbl_id_number = Label(self.frame_right, textvariable=self.id_number)
        self.lbl_id_number.grid(row=0, column=0)

        self.btn_analysis = Button(self.frame_right, text='Analysis', width=20, command=self.analysis)
        self.btn_analysis.grid(row=1, column=0)

        self.pack(fill=BOTH, expand=1)
        self.root.title('Trích xuất thông tin Căn cước công dân')
        self.root.mainloop()

    def open_image(self, event):
        file_name = filedialog.askopenfilename(
            title='Select file',
            filetypes=(('jpeg files', '*.jpg'), ('png files', '*.png'), ('tif files', '*.tif'))
        )

        if file_name:
            self.file_name = file_name
            self.image = PILImage.open(file_name)
            self.image = self.image.resize((400, 400), PILImage.ANTIALIAS)
            self.image = PILPhotoImage(self.image)
            self.canvas.create_image(1, 1, image=self.image, anchor=NW)
            self.id_number.set('')

    def analysis(self):
        if self.file_name:
            id_number = self.id_reader.extract_id_number(self.file_name)
            self.id_number.set(id_number)


if __name__ == '__main__':
    app = Application(Tk())
