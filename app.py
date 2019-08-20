import warnings
import os
from id_card_reader import IdCardReader
from PIL import Image as PILImage
from PIL.ImageTk import PhotoImage as PILPhotoImage
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Application(Frame):
    def __init__(self, root):
        # Initial variables
        self.root = root
        self.image = None
        self.file_name = None
        self.id_reader = IdCardReader()
        self.default_background_image = 'assets/background_400x400.png'

        Frame.__init__(self, self.root, background='white')

        self.frame_left = Frame(self)
        self.frame_left.rowconfigure(1, pad=15)
        self.frame_left.grid(row=0, column=0, sticky=(N, W, E, S))

        self.canvas = Canvas(self.frame_left, highlightthickness=1, highlightbackground='#ccc')
        self.canvas.bind("<ButtonRelease-1>", self.open_image)
        self.canvas.config(width=400, height=400)
        self.canvas.grid(row=0, column=0, padx=(10, 0), pady=10)

        self.image = PILImage.open(self.default_background_image)
        self.image = self.image.resize((400, 400), PILImage.ANTIALIAS)
        self.image = PILPhotoImage(self.image)
        self.canvas.create_image(1, 1, image=self.image, anchor=NW)

        self.frame_right = Frame(self, padx=10, pady=10)
        self.frame_right.rowconfigure(1, pad=15)
        self.frame_right.grid(row=0, column=1, sticky=(N, W, E, S))

        self.lbl1 = Label(self.frame_right, text='Số:')
        self.lbl1.grid(row=0, column=0, sticky=W)

        self.id_number = StringVar()
        self.lbl_id_number = Label(self.frame_right, textvariable=self.id_number, fg='green', font='Helvetica 20 bold')
        self.lbl_id_number.grid(row=0, column=0, padx=(20, 0), pady=(0, 2))

        self.btn_analysis = Button(self.frame_right, text='Phân tích', width=20, command=self.analysis)
        self.btn_analysis.grid(row=1, column=0)

        self.btn_reset = Button(self.frame_right, text='Làm mới', width=20, command=self.reset)
        self.btn_reset.grid(row=2, column=0)

        self.btn_quit = Button(self.frame_right, text='Thoát', width=20, command=self.quit)
        self.btn_quit.grid(row=3, column=0)

        self.pack(fill=BOTH, expand=1)
        self.root.title('Trích xuất thông tin Căn cước công dân')
        self.root.mainloop()

    def open_image(self, event):
        file_name = filedialog.askopenfilename(
            title='Chọn file ảnh',
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
        else:
            messagebox.showinfo('Cảnh báo', 'Bạn chưa chọn ảnh.')

    def reset(self):
        self.image = PILImage.open(self.default_background_image)
        self.image = self.image.resize((400, 400), PILImage.ANTIALIAS)
        self.image = PILPhotoImage(self.image)
        self.canvas.create_image(1, 1, image=self.image, anchor=NW)
        self.file_name = None
        self.id_number.set('')


if __name__ == '__main__':
    app = Application(Tk())
