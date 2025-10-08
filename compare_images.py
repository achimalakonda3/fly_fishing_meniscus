import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import numpy as np
import cv2
import polars as pl
from scipy.optimize import fsolve
import imageio
from tkinter import filedialog

from snells_law_viz import diffraction_differences_to_meniscus

def create_and_warp_speckle_image(width=1920, height=1080):
    """
    Generates a synthetic speckle image and a warped version of it.
    This saves two images: 'speckle_ref.png' and 'speckle_warped.png'.
    """
    print("Generating synthetic speckle images...")
    
    # 1. Create a black reference image
    ref_image = np.zeros((height, width), dtype=np.uint8)
    
    # 2. Add random white speckles
    num_speckles = 600000
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

class Crop_Rotate_ImageCanvas:
    def __init__(self, parent_frame, image_path, scale):
        self.image_path = image_path
        self.scale = scale
        self.current_scale = scale
        self.transformed = False
        self.cropped = False
        self.src_pts = []
        self.image_transformation_matrix = None

        self.pil_image = Image.open(image_path)
        self.orig_width, self.orig_height = self.pil_image.size
        self.aspect_ratio = self.orig_width / self.orig_height
        self.display_size = (int(self.orig_width * scale), int(self.orig_height * scale))
        self.tk_image = None

        self.final_display_size = self.display_size

        self.canvas = tk.Canvas(parent_frame, width = self.display_size[0], height = self.display_size[1])
        self.canvas.pack(side=tk.LEFT)
        self.canvas.focus_set()

        self.load_display_image()

        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<KeyPress>", self.handle_keypress)

        self.on_transform_complete = None
        self.on_commence_comparison = None

    def transform_image(self, image_transformation):
        self.image_mat = np.array(self.pil_image)
        
        output_size = (self.orig_width, self.orig_height) # Use (width, height) tuple
        
        self.transformed_image_mat = cv2.warpAffine(self.image_mat, image_transformation, output_size)
        
        self.image_to_display = Image.fromarray(self.transformed_image_mat)
        
        print(f"Received Transformation Trigger for {self.image_path}")
        self.transformed = True
        self.load_display_image() # This will now resize the new full-res image for display

    def load_display_image(self):
        if not self.transformed:
            self.image_to_display = self.pil_image
            self.current_scale = self.scale
            for i, (x,y) in enumerate(self.src_pts):
                self.canvas.create_oval()
        else:
            self.current_scale = 0.5

        resized_image = self.image_to_display.resize(self.display_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0,0, anchor="nw", image=self.tk_image)

    def handle_click(self, event):
        self.canvas.focus_set()
        if len(self.src_pts) < 2:
            x_display, y_display = event.x, event.y
            self.src_pts.append((x_display, y_display))
            print(f"{np.array(self.src_pts)}")
            print(f"{len(self.src_pts)}")

    def handle_keypress(self, event):
        if event.keysym == 'e':
            print(f"{np.array(self.src_pts)}")
            print(f"{len(self.src_pts)}")
            if len(self.src_pts) == 2 : # Once we have two points selected
                self.image_mat = np.array(self.pil_image)
                self.calculate_image_transformation()
                self.on_transform_complete(self.image_transformation_matrix)

        if event.keysym == 'a':
            self.on_commence_comparison()

    def calculate_image_transformation(self):
        """
        CORRECTED: This function now uses a consistent scale and a much simpler, more
        robust method for defining source and destination points.
        """
        # --- 1. Use the CONSISTENT scale factor ---
        # The scale used to display the image is self.scale. We MUST use this same
        # scale to convert click coordinates.
        # Since your canvas is the same size as the displayed image, there is no offset.
        
        # PROBLEM 1 FIX: Use the known display scale, not a recalculated one.
        display_scale = self.scale

        p1_canvas, p2_canvas = self.src_pts
        src_pts_orig = np.float32([
            [p1_canvas[0] / display_scale, p1_canvas[1] / display_scale],
            [p2_canvas[0] / display_scale, p2_canvas[1] / display_scale]
        ])

        # --- 2. Define Stable Destination Points in the FINAL OUTPUT's coordinate system ---
        # The final output image will have dimensions (self.orig_width, self.orig_height).
        # Let's define our target points within that absolute space.
        
        # PROBLEM 3 FIX: Define destination points simply and absolutely.
        target_center_y = self.orig_height / 2
        # Make the rod occupy 60% of the final image's width
        target_half_width = (self.orig_width * 0.6) / 2
        
        self.dst_pts_orig = np.float32([
            [(self.orig_width / 2) - target_half_width, target_center_y],
            [(self.orig_width / 2) + target_half_width, target_center_y]
        ])
        
        print("Source Points (Original Coords):", src_pts_orig)
        print("Destination Points (Original Coords):", self.dst_pts_orig)

        # --- 3. Let OpenCV Calculate the Matrix ---
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts_orig, self.dst_pts_orig)

        if transformation_matrix is None:
            print("Error: Could not estimate transformation. Check click points are not identical.")
            return

        self.image_transformation_matrix = transformation_matrix
        print("Robust Transformation Matrix:\n", self.image_transformation_matrix)

class Image_Compare_App:
    def __init__(self, root, scale=0.5):
        self.root = root
        self.root.title("Transform + Analyze")

        self.scale = scale
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.canvases = []

        for i in range(2):
            print(f"select image{i + 1}")
            path = filedialog.askopenfilename(
                title=f"Select image {i + 1}",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not path:
                print("Image selection cancelled. Exiting.")
                root.destroy()
                return
            
            image_canvas = Crop_Rotate_ImageCanvas(self.frame, path, scale)
            image_canvas.on_transform_complete = self.sync_transform
            image_canvas.on_commence_comparison = self.generate_comparison_csv
            self.canvases.append(image_canvas)

    def sync_transform(self, image_transformation_matrix):
        for canvas in self.canvases:
            canvas.transform_image(image_transformation_matrix)

    def analyze_warping(self, img1, img2):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        print(f"Found {len(kp1)} keypoints in the first image")
        print(f"Found {len(kp2)} keypoints in the second image")
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        print(f"Found {len(good_matches)} good matches after ratio test")

        if len(good_matches) < 10:
            print("Not enough good matches found")
            return

        # Convert images to color if needed (in case they're grayscale)
        if len(img1.shape) == 2 or img1.shape[2] == 1:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()

        if len(img2.shape) == 2 or img2.shape[2] == 1:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()

        x_y_diffs_temp = []
        image_2_points = []

        for match in good_matches:
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)
            image_2_points.append(pt2)
            x_y_diffs_temp.append(pt1 - pt2)

        image_2_points = np.array(image_2_points)
        x_y_diffs_temp = np.array(x_y_diffs_temp)
        x_y_diffs = x_y_diffs_temp[np.linalg.norm(x_y_diffs_temp[:, :2], axis=1) <= 50]
        image_2_points = image_2_points[np.linalg.norm(x_y_diffs_temp[:, :2], axis=1) <= 50]

        # Plot the difference vectors (x and y components separately)
        plt.plot(x_y_diffs[:, 0], label='x diff')
        plt.plot(x_y_diffs[:, 1], label='y diff')
        plt.legend()
        plt.title("Displacement Vectors (x and y)")
        plt.show()

        x = image_2_points[:, 0]
        y = image_2_points[:, 1]

        fig, ax = plt.subplots()
        ax.imshow(img2)
        ax.quiver(x, y, x_y_diffs[:, 0], x_y_diffs[:, 1],
                angles="xy", scale_units='xy', scale=1, color='red', width=0.003)
        ax.set_title("Displacement Field")
        plt.show()

        return x_y_diffs, image_2_points

    def generate_comparison_csv(self):
        try:
            try:
                img1 = self.canvases[0].transformed_image_mat
                img2 = self.canvases[1].transformed_image_mat
                x_y_diffs, img2_pts = self.analyze_warping(img1, img2)
                data_to_save = np.hstack((x_y_diffs, img2_pts))
            except:
                print("Error with Analysis")
            
            file_path = filedialog.asksaveasfilename(
                title="Save csv file as",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )


            if file_path:
                with open(file_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['x diff', 'y diff', 'init x', 'init y'])
                    writer.writerows(data_to_save)

        except:
            print("Error with Saving")
        try:
            diffraction_differences_to_meniscus()
        except:
            print("Error Analyzing Data")


    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Image_Compare_App(root, scale = 0.5)
    root.mainloop()

    # # create_and_warp_speckle_image(2550, 3300)