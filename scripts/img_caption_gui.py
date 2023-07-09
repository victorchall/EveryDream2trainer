# Python GUI tool to manually caption images for machine learning.
# A sidecar file is created for each image with the same name and a .txt extension.
#
# [control/command + o] to open a folder of images.
# [page down] and [page up] to go to next and previous images. Hold shift to skip 10 images.
# [shift + home] and [shift + end] to go to first and last images.
# [shift + delete] to move the current image into a '_deleted' folder.
# [escape] to exit the app.

import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path

IMG_EXT = ["jpg", "jpeg", "png", "webp"]

class CaptionedImage():
    def __init__(self, image_path):
        self.base_path = image_path.parent
        self.path = image_path
    
    def caption_path(self):
        return self.base_path / (self.path.stem + '.txt')

    def read_caption(self):
        caption_path = self.caption_path()
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8', newline='') as f:
                return f.read()
        return ''

    def write_caption(self, caption):
        caption_path = self.caption_path()
        with open(str(caption_path), 'w', encoding='utf-8', newline='') as f:
            f.write(caption)
    
    # sort
    def __lt__(self, other):
        return str(self.path).lower() < str(other.path).lower()

class ImageView(tk.Frame):

    def __init__(self, root):
        tk.Frame.__init__(self, root)

        self.root = root
        self.base_path = None
        self.images = []
        self.index = 0

        # image
        self.image_frame = tk.Frame(self)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.image_frame.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        
        # caption field
        self.caption_frame = tk.Frame(self)
        self.caption_field = tk.Text(self.caption_frame, wrap="word", width=50)
        self.caption_field.pack(expand=True, fill=tk.BOTH)
        self.caption_frame.pack(fill=tk.Y, side=tk.RIGHT)

    def open_folder(self):
        dir = filedialog.askdirectory()
        if not dir:
            return
        self.base_path = Path(dir)
        if self.base_path is None:
            return
        self.images.clear()
        for ext in IMG_EXT:
            for file in self.base_path.glob(f'*.{ext}'):
                self.images.append(CaptionedImage(file))
        self.images.sort()
        self.update_ui()
    
    def store_caption(self):
        txt = self.caption_field.get(1.0, tk.END)
        txt = txt.replace('\r', '').replace('\n', '').strip()
        self.images[self.index].write_caption(txt)
        
    def set_index(self, index):
        self.index = index % len(self.images)

    def go_to_image(self, index):
        if len(self.images) == 0:
            return
        self.store_caption()
        self.set_index(index)
        self.update_ui()

    def next_image(self):
        self.go_to_image(self.index + 1)

    def prev_image(self):
        self.go_to_image(self.index - 1)

    # move current image to a "_deleted" folder
    def delete_image(self):
        if len(self.images) == 0:
            return
        img = self.images[self.index]

        trash_path = self.base_path / '_deleted'
        if not trash_path.exists():
            trash_path.mkdir()
        img.path.rename(trash_path / img.path.name)
        caption_path = img.caption_path()
        if caption_path.exists():
            caption_path.rename(trash_path / caption_path.name)
        del self.images[self.index]
        self.set_index(self.index)
        self.update_ui()
    
    def update_ui(self):
        if (len(self.images)) == 0:
            self.filename.set('')
            self.caption_field.delete(1.0, tk.END)
            self.image_label.configure(image=None)
            return
        img = self.images[self.index]
        # filename
        title = self.images[self.index].path.name if len(self.images) > 0 else ''
        self.root.title(title + f' ({self.index+1}/{len(self.images)})')
        # caption
        self.caption_field.delete(1.0, tk.END)
        self.caption_field.insert(tk.END, img.read_caption())
        # image
        img = Image.open(self.images[self.index].path)
        
        # scale the image to fit inside the frame
        w = self.image_frame.winfo_width()
        h = self.image_frame.winfo_height()
        if img.width > w or img.height > h:
            img.thumbnail((w, h))
        photoImage = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photoImage)
        self.image_label.image = photoImage
    
if __name__=='__main__':
    root = tk.Tk()
    root.geometry('1200x800')
    root.title('Image Captions')

    if sys.platform == 'darwin':
        root.bind('<Command-o>', lambda e: view.open_folder())
    else:
        root.bind('<Control-o>', lambda e: view.open_folder())
    root.bind('<Escape>', lambda e: root.destroy())
    root.bind('<Prior>', lambda e: view.prev_image())
    root.bind('<Next>', lambda e: view.next_image())
    root.bind('<Shift-Prior>', lambda e: view.go_to_image(view.index - 10))
    root.bind('<Shift-Next>', lambda e: view.go_to_image(view.index + 10))
    root.bind('<Shift-Home>', lambda e: view.go_to_image(0))
    root.bind('<Shift-End>', lambda e: view.go_to_image(len(view.images) - 1))
    root.bind('<Shift-Delete>', lambda e: view.delete_image())

    view = ImageView(root)
    view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()