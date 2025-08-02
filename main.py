from compare_images import Image_Compare_App
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    app = Image_Compare_App(root, scale = 0.15)
    root.mainloop()
