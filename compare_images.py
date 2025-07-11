import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
import numpy as np
import cv2


class ImageCanvas:
    def __init__(self, parent_frame, image_path, scale):
        self.image_path = image_path
        self.scale = scale
        self.current_scale = scale 
        self.zoomed_in = False
        self.crop_box = None
        self.annotations = []
        self.annotation_array = np.array([])  # Filled before analysis
        self.selected_annotation = None

        self.pil_image = Image.open(image_path)
        self.orig_width, self.orig_height = self.pil_image.size
        self.aspect_ratio = self.orig_width / self.orig_height
        self.display_size = (int(self.orig_width * scale), int(self.orig_height * scale))
        self.tk_image = None

        self.canvas = tk.Canvas(parent_frame, width=self.display_size[0], height=self.display_size[1])
        self.canvas.pack(side=tk.LEFT)
        self.canvas.focus_set()

        self.rect = None
        self.start_x = self.start_y = None

        self.load_display_image()

        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        self.canvas.bind("<KeyPress>", self.move_annotation)

        self.on_crop_complete = None

    def load_display_image(self):
        if self.zoomed_in and self.crop_box:
            # We are zoomed in. Crop the original image and prepare for display.
            crop_x1, crop_y1, _, _ = self.crop_box
            image_to_display = self.pil_image.crop(self.crop_box)
            self.current_scale = self.display_size[0] / image_to_display.width
        else:
            # We are in the default, full view.
            crop_x1, crop_y1 = 0, 0
            image_to_display = self.pil_image
            self.current_scale = self.scale

        # This part remains the same
        resized_image = image_to_display.resize(self.display_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Correctly draw annotations based on the current view
        for i, (x, y) in enumerate(self.annotations):
            # Translate original coordinates to the coordinate system of the cropped view
            display_x = (x - crop_x1) * self.current_scale
            display_y = (y - crop_y1) * self.current_scale
            
            # Check if the annotation is within the current crop_box to avoid drawing outside
            if self.zoomed_in and not (self.crop_box[0] <= x < self.crop_box[2] and self.crop_box[1] <= y < self.crop_box[3]):
                continue

            color = 'blue' if self.selected_annotation == i else 'red'
            r = 3
            self.canvas.create_oval(
                display_x - r, display_y - r, display_x + r, display_y + r,
                fill=color)
            self.canvas.create_text(
                display_x + 10, display_y, anchor='w',
                text=f"({x}, {y})", fill=color)

    def handle_click(self, event):
        if not self.zoomed_in:
            self.start_x, self.start_y = event.x, event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
        else:
            x_display, y_display = event.x, event.y
            crop_x1, crop_y1 = self.crop_box[0], self.crop_box[1]
            x_orig = int(x_display / self.current_scale) + crop_x1
            y_orig = int(y_display / self.current_scale) + crop_y1

            # Check if user clicked close to an existing annotation to select it
            for i, (ax, ay) in enumerate(self.annotations):
                if abs(ax - x_orig) <= 3 and abs(ay - y_orig) <= 3:
                    self.selected_annotation = i
                    self.load_display_image()
                    return

            self.annotations.append((x_orig, y_orig))
            self.selected_annotation = len(self.annotations) - 1
            self.load_display_image()

    def update_crop(self, event):
        if self.zoomed_in or not self.rect:
            return  
        
        # Assume self.aspect_ratio is defined (e.g., in __init__ as self.aspect_ratio = 16/9)
        end_x, end_y = event.x, event.y
        width = end_x - self.start_x
        height = end_y - self.start_y

        # Adjust the height based on the width to enforce the aspect ratio
        # This preserves the direction of the drag (e.g., up-left, down-right)
        if height >= 0:
            end_y = self.start_y + abs(width) / self.aspect_ratio
        else:
            end_y = self.start_y - abs(width) / self.aspect_ratio
        
        self.canvas.coords(self.rect, self.start_x, self.start_y, end_x, end_y)
        

    def end_crop(self, event):
        if self.zoomed_in:
            return

        end_x, end_y = event.x, event.y
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)

        self.canvas.delete(self.rect)
        self.rect = None

        x1_orig = int(x1 / self.scale)
        y1_orig = int(y1 / self.scale)
        x2_orig = int(x2 / self.scale)
        y2_orig = int(y2 / self.scale)

        self.crop_box = (x1_orig, y1_orig, x2_orig, y2_orig)
        if self.on_crop_complete:
            self.on_crop_complete(self.crop_box)

    def zoom_to_crop(self, crop_box):
        self.crop_box = crop_box
        cropped = self.pil_image.crop(crop_box)
        self.load_display_image()
        self.zoomed_in = True

        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

    def move_annotation(self, event):
        if self.selected_annotation is None:
            return

        x, y = self.annotations[self.selected_annotation]
        if event.keysym == 'Left':
            x -= 1
        elif event.keysym == 'Right':
            x += 1
        elif event.keysym == 'Up':
            y -= 1
        elif event.keysym == 'Down':
            y += 1

        self.annotations[self.selected_annotation] = (x, y)
        self.load_display_image()

    def export_annotations(self):
        self.annotation_array = np.array(self.annotations)

        if len(self.annotations) == 0:
            print(f"No annotations for {self.image_path}")
            return

        base = os.path.basename(self.image_path)
        name, _ = os.path.splitext(base)
        out_file = f"annotations_{name}.csv"

        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(self.annotations)

        print(f"✅ Exported annotations for {name} to '{out_file}'")


def analyze_points(arrays: list[np.ndarray]):
    """
    Your data analysis logic goes here.
    Input: a list of NumPy arrays (one per image), shape (N, 2)
    Example: compute centroid, distances, correlations, etc.
    """
    print("\n📊 Running analysis on annotations...\n")

    for i, arr in enumerate(arrays):
        print(f"Image {i + 1}: {arr.shape[0]} points")
        if arr.shape[0] > 0:
            centroid = arr.mean(axis=0)
            print(f"  ↪ Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        else:
            print("  ↪ No points to analyze.")


class ImageClickApp:
    def __init__(self, root, scale=0.5):
        self.root = root
        self.root.title("Zoom + Annotate + Analyze")

        self.scale = scale
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.canvases = []

        for i in range(2):
            print(f"Select image {i + 1}")
            path = filedialog.askopenfilename(
                title=f"Select image {i + 1}",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not path:
                print("Image selection cancelled. Exiting.")
                root.destroy()
                return

            image_canvas = ImageCanvas(self.frame, path, scale)
            image_canvas.on_crop_complete = self.sync_zoom
            self.canvases.append(image_canvas)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def sync_zoom(self, crop_box):
        for canvas in self.canvases:
            canvas.zoom_to_crop(crop_box)

    def on_close(self):
        for canvas in self.canvases:
            canvas.export_annotations()

        # Call your analysis function
        analyze_points([c.annotation_array for c in self.canvases])

        self.root.destroy()


def create_and_warp_speckle_image(width=1920, height=1080):
    """
    Generates a synthetic speckle image and a warped version of it.
    This saves two images: 'speckle_ref.png' and 'speckle_warped.png'.
    """
    print("Generating synthetic speckle images...")
    
    # 1. Create a black reference image
    ref_image = np.zeros((height, width), dtype=np.uint8)
    
    # 2. Add random white speckles
    num_speckles = 60000
    xs = np.random.randint(0, width, num_speckles)
    ys = np.random.randint(0, height, num_speckles)
    for x, y in zip(xs, ys):
        cv2.circle(ref_image, (x, y), radius=2 , color=(255, 255, 255), thickness=-1)
        
    cv2.imwrite('speckle_ref.png', ref_image)
    print("Saved 'speckle_ref.png'")

    # 3. Define a warp (e.g., rotation and slight scaling/shear)
    center = (width // 2, height // 2)
    angle = 5.0  # degrees
    scale = 1.05
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Add a small translation (shift)
    M[0, 2] += 15 # x-shift
    M[1, 2] += 10 # y-shift

    # 4. Apply the affine warp
    warped_image = cv2.warpAffine(ref_image, M, (width, height))
    cv2.imwrite('speckle_warped.png', warped_image)
    print("Saved 'speckle_warped.png'")
    
    return ref_image, warped_image

def visualize_warping(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"Found {len(kp1)} keypoints in the first image")
    print(f"Found {len(kp2)} keypoints in the second image")
    
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        # m is the best match, n is the second-best match
        # If the distance to the best match is significantly smaller than the second best, then the match is considered confident
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

        print(f"Found {len(good_matches)} good matches after ratio test")

    if len(good_matches) < 10:
        print("Not enough good matches found")
        exit()
    
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAYBGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAYBGR)
    vis = np.hstack((img1_color, img2_color))

    pt1 = (int(kp1[pt1_idx].pt[0]), int(kp1[pt1_idx].pt[1]))
    pt2 = (int(kp2[pt2_idx].pt[0]), int(kp2[pt2_idx].pt[1]))

    pt2_shifted = (pt2[0] + img1.shape[1], pt2[1])

    color = np.random.randint(0,255,(3,)).to_list()
    

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClickApp(root, scale=0.15)
    root.mainloop()

    # create_and_warp_speckle_image()