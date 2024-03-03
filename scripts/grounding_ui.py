import tkinter as tk
from tkinter import filedialog, Listbox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
import ast

class ImageBoundingBoxApp:
    def __init__(self, root, folder_path):
        self.root = root
        self.folder_path = folder_path
        self.images = self.load_images(folder_path)
        self.setup_ui()

    def load_images(self, folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.webp'))]
        return images

    def setup_ui(self):
        self.listbox = Listbox(self.root)
        self.listbox.pack(side="left", fill="y")

        self.canvas = tk.Canvas(self.root, bg='white')
        self.canvas.pack(side="right", expand=True, fill="both")

        for image in self.images:
            self.listbox.insert(tk.END, image)

        self.listbox.bind("<<ListboxSelect>>", self.on_select)

    def on_select(self, event):
        w = event.widget
        index = int(w.curselection()[0])
        image_name = w.get(index)
        self.display_image_with_bboxes(image_name)

    def display_image_with_bboxes(self, image_name):
        img_path = os.path.join(self.folder_path, image_name)
        ent_path = img_path.rsplit('.', 1)[0] + '.ent'

        # Load image
        image = cv2.imread(img_path)
        image_h, image_w, _ = image.shape
        font_scale = max(0.5 * (image_h + image_w) / 1000,0.35)
        thickness = max(int(0.5 * (image_h + image_w) / 500),1)
        print(f"Font Scale: {font_scale}, Thickness: {thickness}")

        # Read .ent file
        if os.path.exists(ent_path):
            with open(ent_path, 'r') as f:
                entities = ast.literal_eval(f.read())

            for entity_name, _, bboxes in entities:
                for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
                    orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)
                    #shadow text
                    cv2.putText(image, entity_name, (orig_x1, orig_y1 + int(30*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+3)
                    #cv2.putText(image, entity_name, (orig_x1+1, orig_y1 + int(30*font_scale)-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
                    #text
                    cv2.putText(image, entity_name, (orig_x1, orig_y1 + int(30*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

        # Convert to PIL Image to display in Tkinter
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Resize image to fit the canvas size while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = image_pil.size

        # Calculate the proper scaling factor
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h)

        # Calculate new image size
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize the image with new dimensions
        image_resized = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_resized)

        # Clear the canvas and display the resized image
        self.canvas.delete("all")  # Clear previous image/text
        self.canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor="nw", image=image_tk)
        self.canvas.image = image_tk  # Keep a reference!

def main():
    root = tk.Tk()
    root.title("Image Bounding Box Viewer")

    folder_path = filedialog.askdirectory(title="Select Folder with Images")
    app = ImageBoundingBoxApp(root, folder_path)
    
    root.mainloop()

if __name__ == "__main__":
    main()
