import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import csv
import numpy as np
import cv2
import polars as pl
from scipy.optimize import fsolve
import imageio
from io import StringIO
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Placeholder for the external analysis function ---
def diffraction_differences_to_meniscus():
    """
    Placeholder function for further data analysis.
    """
    st.info("Placeholder: Further analysis using 'diffraction_differences_to_meniscus' would be performed here.")
    print("Executing placeholder for diffraction_differences_to_meniscus.")


# --- Core Image Processing and Analysis Functions (Unchanged) ---

def create_and_warp_speckle_image(width=1920, height=1080):
    """
    Generates a synthetic speckle image and a warped version of it.
    """
    st.write("Generating synthetic speckle images...")
    ref_image = np.zeros((height, width), dtype=np.uint8)
    num_speckles = 600000
    xs = np.random.randint(0, width, num_speckles)
    ys = np.random.randint(0, height, num_speckles)
    for x, y in zip(xs, ys):
        cv2.circle(ref_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)

    center = (width // 2, height // 2)
    angle = 5.0
    scale = 1.05
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 15
    M[1, 2] += 10
    warped_image = cv2.warpAffine(ref_image, M, (width, height))
    st.success("Generated synthetic speckle images in memory.")
    return ref_image, warped_image

def calculate_image_transformation(src_pts, orig_width, orig_height):
    """
    Calculates the affine transformation matrix to align the selected points.
    """
    p1_canvas, p2_canvas = src_pts
    src_pts_orig = np.float32([p1_canvas, p2_canvas])

    target_center_y = orig_height / 2
    target_half_width = (orig_width * 0.6) / 2
    
    dst_pts_orig = np.float32([
        [(orig_width / 2) - target_half_width, target_center_y],
        [(orig_width / 2) + target_half_width, target_center_y]
    ])
    
    transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts_orig, dst_pts_orig)
    return transformation_matrix

def analyze_warping(img1, img2):
    """
    Analyzes the warping between two images using SIFT feature matching.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    st.write(f"Found {len(kp1)} keypoints in image 1 and {len(kp2)} in image 2.")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    st.write(f"Found {len(good_matches)} good matches after ratio test.")

    if len(good_matches) < 10:
        st.warning("Not enough good matches found to perform analysis.")
        return None, None, None, None

    x_y_diffs_temp, image_2_points_temp = [], []
    for match in good_matches:
        pt1 = np.array(kp1[match.queryIdx].pt)
        pt2 = np.array(kp2[match.trainIdx].pt)
        image_2_points_temp.append(pt2)
        x_y_diffs_temp.append(pt1 - pt2)

    image_2_points, x_y_diffs_temp = np.array(image_2_points_temp), np.array(x_y_diffs_temp)
    
    mask = np.linalg.norm(x_y_diffs_temp[:, :2], axis=1) <= 50
    x_y_diffs, image_2_points = x_y_diffs_temp[mask], image_2_points[mask]

    fig1, ax1 = plt.subplots()
    ax1.plot(x_y_diffs[:, 0], label='x diff')
    ax1.plot(x_y_diffs[:, 1], label='y diff')
    ax1.legend()
    ax1.set_title("Displacement Vectors (x and y)")

    fig2, ax2 = plt.subplots()
    ax2.imshow(img2, cmap='gray')
    ax2.quiver(image_2_points[:, 0], image_2_points[:, 1], x_y_diffs[:, 0], x_y_diffs[:, 1],
               angles="xy", scale_units='xy', scale=1, color='red', width=0.003)
    ax2.set_title("Displacement Field")

    return x_y_diffs, image_2_points, fig1, fig2

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("Image Transformation and Warping Analysis App")

# --- Initialize Session State ---
def initialize_state():
    defaults = {
        'points': [], 'img1_orig': None, 'img2_orig': None,
        'img1_transformed': None, 'img2_transformed': None,
        'analysis_data': None, 'plot1': None, 'plot2': None,
        'last_coord_1': None, 'last_coord_2': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
initialize_state()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    display_width = st.slider("Image Display Width (pixels)", 200, 1000, 500)
    
    uploaded_file1 = st.file_uploader("Choose the first image", type=["png", "jpg", "jpeg"])
    if uploaded_file1:
        st.session_state.img1_orig = Image.open(uploaded_file1)

    uploaded_file2 = st.file_uploader("Choose the second image", type=["png", "jpg", "jpeg"])
    if uploaded_file2:
        st.session_state.img2_orig = Image.open(uploaded_file2)
        
    if st.button("Generate Synthetic Speckle Images"):
        ref_img_data, warped_img_data = create_and_warp_speckle_image()
        st.session_state.img1_orig = Image.fromarray(ref_img_data)
        st.session_state.img2_orig = Image.fromarray(warped_img_data)
        st.session_state.points = [] # Reset points
        st.rerun()

# --- Main App Body ---
if not st.session_state.img1_orig or not st.session_state.img2_orig:
    st.info("Please upload two images or generate synthetic ones using the sidebar to begin.")
else:
    col1, col2 = st.columns(2)
    new_point = None
    
    # --- Image Display and Point Selection ---
    
    # Resize images for display and calculate scaling factors
    w1, h1 = st.session_state.img1_orig.size
    scale1 = display_width / w1
    display_img1 = st.session_state.img1_orig.resize(
        (display_width, int(h1 * scale1)), Image.Resampling.LANCZOS
    )
    
    w2, h2 = st.session_state.img2_orig.size
    scale2 = display_width / w2
    display_img2 = st.session_state.img2_orig.resize(
        (display_width, int(h2 * scale2)), Image.Resampling.LANCZOS
    )

    with col1:
        st.header("Image 1 (Click to select)")
        coord1 = streamlit_image_coordinates(display_img1, key="img1_coords")
        if coord1 and coord1 != st.session_state.last_coord_1:
            st.session_state.last_coord_1 = coord1
            # Scale coordinates back to original image size
            original_x = coord1['x'] / scale1
            original_y = coord1['y'] / scale1
            new_point = (original_x, original_y)

    with col2:
        st.header("Image 2 (Click to select)")
        coord2 = streamlit_image_coordinates(display_img2, key="img2_coords")
        if coord2 and coord2 != st.session_state.last_coord_2:
            st.session_state.last_coord_2 = coord2
            # Scale coordinates back to original image size
            original_x = coord2['x'] / scale2
            original_y = coord2['y'] / scale2
            new_point = (original_x, original_y)

    # Add a newly clicked point if there's space
    if new_point and len(st.session_state.points) < 2:
        st.session_state.points.append(new_point)
        st.rerun()

    st.header("Transformation Controls")
    # Display points rounded to nearest integer for clarity
    st.write("Selected Points (Original Coords):", [ (round(p[0]), round(p[1])) for p in st.session_state.points])

    if st.button("Reset Selection"):
        st.session_state.points = []
        st.session_state.last_coord_1 = None
        st.session_state.last_coord_2 = None
        st.rerun()

    # --- Transformation ---
    if len(st.session_state.points) == 2:
        if st.button("Calculate and Apply Transformation"):
            img1_np = np.array(st.session_state.img1_orig.convert('L'))
            img2_np = np.array(st.session_state.img2_orig.convert('L'))
            h, w = img1_np.shape[:2]

            trans_matrix = calculate_image_transformation(st.session_state.points, w, h)
            
            if trans_matrix is not None:
                st.session_state.img1_transformed = cv2.warpAffine(img1_np, trans_matrix, (w, h))
                st.session_state.img2_transformed = cv2.warpAffine(img2_np, trans_matrix, (w, h))
                st.success("Transformation applied successfully!")
            else:
                st.error("Could not estimate transformation. Try selecting different points.")

    # --- Analysis ---
    if st.session_state.img1_transformed is not None:
        st.header("Transformed Images")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.image(st.session_state.img1_transformed, caption="Transformed Image 1", width=display_width)
        with t_col2:
            st.image(st.session_state.img2_transformed, caption="Transformed Image 2", width=display_width)

        st.header("Analysis")
        if st.button("Analyze Warping and Generate CSV"):
            with st.spinner("Analyzing images..."):
                diffs, pts, fig1, fig2 = analyze_warping(st.session_state.img1_transformed, st.session_state.img2_transformed)
                if diffs is not None:
                    st.session_state.analysis_data = np.hstack((diffs, pts))
                    st.session_state.plot1, st.session_state.plot2 = fig1, fig2
                    st.success("Analysis complete.")
                else:
                    st.session_state.analysis_data = st.session_state.plot1 = st.session_state.plot2 = None
                    
    # --- Results ---
    if st.session_state.analysis_data is not None:
        st.header("Analysis Results")
        
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['x diff', 'y diff', 'init x', 'init y'])
        writer.writerows(st.session_state.analysis_data)
        
        st.download_button("Download Analysis as CSV", csv_buffer.getvalue(), "displacement_analysis.csv", "text/csv")
        
        diffraction_differences_to_meniscus()

        st.pyplot(st.session_state.plot1)
        st.pyplot(st.session_state.plot2)