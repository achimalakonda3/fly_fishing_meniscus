import tkinter as tk
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
import numpy as np
import cv2

class Crop_and_Point_Clicking_ImageCanvas:
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

            image_canvas = Crop_and_Point_Clicking_ImageCanvas(self.frame, path, scale)
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

def analyze_points():
    """
    Subtract the point positions from identical ones
    """
    print("\n📊 Running analysis on annotations...\n")

    root = tk.Tk()
    csv_file_path_1 = filedialog.askopenfilename(
        title = "Select CSV File 1",
        filetypes = [("CSV Files", "*.csv")]
    )
    csv_file_path_2 = filedialog.askopenfilename(
        title = "Select CSV File 2",
        filetypes = [("CSV Files", "*.csv")]
    )

    image_1_points = np.genfromtxt(csv_file_path_1, delimiter=',', skip_header=1)
    image_2_points = np.genfromtxt(csv_file_path_2, delimiter=',', skip_header=1)
    image_1_points = np.delete(image_1_points, (0), axis = 0)
    image_2_points = np.delete(image_2_points, (0), axis = 0)
    image_2_points = np.delete(image_2_points, (55), axis = 0)

    x_y_diffs = image_1_points - image_2_points[:len(image_1_points)]
    
    specimen_image_path = filedialog.askopenfilename(
                title=f"Select image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
    specimen_image = Image.open(specimen_image_path)
    # Image Size
    height, width = specimen_image.size[1], specimen_image.size[0]

    plt.plot(x_y_diffs)
    plt.legend(["x", "y"])
    plt.show()
    x = image_2_points[:,0]
    y = image_2_points[:,1]
    grid_x, grid_y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    heatmap_interp = griddata(
        points = (x,y),
        values=x_y_diffs,
        xi=(grid_x, grid_y),
        method='linear',
        fill_value=0
    )

    fig, ax = plt.subplots()
    ax.imshow(specimen_image)
    ax.quiver(
        x,y,x_y_diffs[:,0], x_y_diffs[:,1],
        angles="xy",
        scale_units = 'xy',
        scale = 1,
        color = 'red',
        width = 0.005
    )
    plt.show()

    root.destroy()

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

        self.final_display_size = (int(400), int(600)) # width , height

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
        self.transformed_image_mat = cv2.warpAffine(self.image_mat, image_transformation, self.final_display_size)
        self.image_to_display = Image.fromarray(self.transformed_image_mat)
        print(f"Received Transformation Trigger for {self.image_path}")
        self.transformed = True
        self.load_display_image()

    def load_display_image(self):
        if not self.transformed:
            self.image_to_display = self.pil_image
            self.current_scale = self.scale
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
        # Get original image and canvas dimensions
        orig_h, orig_w = self.image_mat.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Calculate the single, aspect-ratio-preserving scale factor
        scale = min(canvas_w / orig_w, canvas_h / orig_h)

        # Calculate the size of the image as it's actually displayed on the canvas
        # This is NOT the same as canvas_w, canvas_h if there's letterboxing
        display_w = int(orig_w * scale)
        display_h = int(orig_h * scale)

        # Calculate the padding (offset) of the image from the top-left of the canvas
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2

        # Now, when converting click points, use this scale and offset
        p1_canvas, p2_canvas = self.src_pts
        src_pts_orig = np.float32([
            [(p1_canvas[0] - offset_x) / scale, (p1_canvas[1] - offset_y) / scale],
            [(p2_canvas[0] - offset_x) / scale, (p2_canvas[1] - offset_y) / scale]
        ])

        # 2. Define Destination Points in the Original Image's Coordinate Space
        # We want the rod to be centered horizontally on the canvas.
        # Let's make it occupy 80% of the canvas width.
        canvas_center_x = (self.canvas.winfo_width() / 2) * scale
        canvas_center_y = (self.canvas.winfo_height() / 2) * scale
        target_half_width = (self.canvas.winfo_width() * 0.8 / 2) * scale

        dst_pts_orig = np.float32([
            [canvas_center_x - target_half_width, canvas_center_y],
            [canvas_center_x + target_half_width, canvas_center_y]
        ])

        # 3. Let OpenCV Calculate the Best Transformation Matrix
        # This function finds the optimal rotation, translation, and uniform scaling.
        # It's perfect for this use case and handles all the math internally.
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts_orig, dst_pts_orig)

        # The result is a 2x3 matrix, exactly what warpAffine needs.
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
            image_canvas.on_commence_comparison = self.analyze_warping
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
        ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
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
            
            file_object = filedialog.asksaveasfile(title="Save csv file as", mode='w', defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_object:
                with file_object as f:
                    writer = csv.writer(f)
                    writer.writerow(['x diff', 'y diff', 'init x', 'init y'])
                    writer.writerows(data_to_save)
        except:
            print("Error with Saving")

    def on_close(self):
        

        


def visualize_warping(image_paths):
    img1 = cv2.imread(image_paths[0])
    img2 = cv2.imread(image_paths[1])
    img1 = img1[1250:1750, 2700:3300]
    img2 = img2[1250:1750, 2700:3300]
    
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

    # Stack images horizontally for visualization (not used here)
    # vis = np.hstack((img1_color, img2_color))

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
    ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax.quiver(x, y, x_y_diffs[:, 0], x_y_diffs[:, 1],
              angles="xy", scale_units='xy', scale=1, color='red', width=0.003)
    ax.set_title("Displacement Field")
    plt.show()

    return x_y_diffs, image_2_points

def crop_and_show_images():
    # Hide the main Tkinter window
    root = tk.Tk()
    root.withdraw()

    image_paths = []
    for _ in range(2):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        image_paths.append(file_path)

        # Proceed only if a file was selected
        if file_path:
            # Load the image
            image = cv2.imread(file_path)

            # Crop the image (y1:y2, x1:x2)
            cropped = image[1250:1750, 2700:3300]  # Adjust as needed

            # Display the cropped image
            cv2.imshow("Cropped Image", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No file selected.")
    
    return image_paths

if __name__ == "__main__":
    # root = tk.Tk()
    # app = ImageClickApp(root, scale=0.15)
    # root.mainloop()

    root = tk.Tk()
    app = Image_Compare_App(root, scale = 0.15)
    root.mainloop()

    # # create_and_warp_speckle_image(2550, 3300)
    # image_paths = crop_and_show_images()
    # x_y_diffs, image_2_points = visualize_warping(image_paths)
    
    # analyze_points()