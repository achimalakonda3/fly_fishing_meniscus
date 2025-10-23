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
from sklearn.neighbors import NearestNeighbors # IMPORTED FOR OUTLIER REMOVAL

# This function performs calculations and returns the final DataFrame and a plot figure
def diffraction_differences_to_meniscus(diffraction_data_df, orig_width, depth = 14/32 * 23.5, water_RI = 1.3333, length = 20):
    """
    Takes a Polars DataFrame with raw displacement data, calculates meniscus
    angles based on Snell's Law, and returns an enriched DataFrame and a plot.
    """
    # MODIFIED: mm_per_px is now calculated dynamically based on the image width and the 60% target width from transformation
    pixels_for_length = orig_width * 0.6
    mm_per_px = length / pixels_for_length
    
    # Perform calculations using Polars expressions
    diffraction_data_df = diffraction_data_df.with_columns(
        (pl.arctan2(pl.col("y diff") * mm_per_px, pl.lit(depth))).alias("beta"),
        (pl.col("init x") * mm_per_px).alias("init x (mm)"),
        (pl.col("init y") * mm_per_px).alias("init y (mm)"),
    )

    betas = diffraction_data_df["beta"].to_numpy()

    def snells_law_eqns(thetas, beta):
        return [thetas[0] + beta - thetas[1], 
                np.sin(thetas[0]) - water_RI * np.sin(thetas[1])]
    
    theta_1s, theta_2s = [], []

    for b in betas:
        soln = fsolve(snells_law_eqns, x0=[0,0], args=(b,))
        theta_1s.append(soln[0])
        theta_2s.append(soln[1])
    
    diffraction_data_df = diffraction_data_df.with_columns([
        pl.Series("theta 1 (rad)", theta_1s),
        pl.Series("theta 2 (rad)", theta_2s),
        pl.Series("theta 1 (deg)", np.degrees(theta_1s)),
        pl.Series("theta 2 (deg)", np.degrees(theta_2s)),
    ])

    fig, ax = plt.subplots()
    ax.plot(diffraction_data_df["init y (mm)"], diffraction_data_df["theta 1 (deg)"], '.b')
    ax.set_xlabel("Y Position (mm)")
    ax.set_ylabel("Water Surface Angle (degrees)")
    ax.set_title("Y Position vs Water Surface Angle")
    
    return diffraction_data_df, fig

# --- Core Image Processing and Analysis Functions ---
def create_and_warp_speckle_image(width=1920, height=1080):
    st.write("Generating synthetic speckle images...")
    ref_image = np.zeros((height, width), dtype=np.uint8)
    num_speckles = 600000
    xs, ys = np.random.randint(0, width, num_speckles), np.random.randint(0, height, num_speckles)
    for x, y in zip(xs, ys):
        cv2.circle(ref_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
    center = (width // 2, height // 2)
    angle, scale = 5.0, 1.05
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 15
    M[1, 2] += 10
    warped_image = cv2.warpAffine(ref_image, M, (width, height))
    st.success("Generated synthetic speckle images in memory.")
    return ref_image, warped_image

def calculate_image_transformation(src_pts, orig_width, orig_height):
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

# MODIFIED: This function now includes outlier removal
def analyze_warping(img1, img2):
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
        pt1, pt2 = np.array(kp1[match.queryIdx].pt), np.array(kp2[match.trainIdx].pt)
        image_2_points_temp.append(pt2)
        x_y_diffs_temp.append(pt1 - pt2)
    image_2_points, x_y_diffs_temp = np.array(image_2_points_temp), np.array(x_y_diffs_temp)
    mask = np.linalg.norm(x_y_diffs_temp[:, :2], axis=1) <= 50
    x_y_diffs, image_2_points = x_y_diffs_temp[mask], image_2_points[mask]

    # --- NEW: OUTLIER REMOVAL LOGIC ---
    if len(x_y_diffs) > 5: # Need at least 6 points to have 5 neighbors
        st.write("Performing outlier removal based on local vector magnitudes...")
        magnitudes = np.linalg.norm(x_y_diffs, axis=1)
        
        # Find 5 nearest neighbors for each point (k=6 because a point is its own neighbor)
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(image_2_points)
        distances, indices = nbrs.kneighbors(image_2_points)
        
        # Identify outliers
        is_outlier = np.zeros(len(magnitudes), dtype=bool)
        for i in range(len(magnitudes)):
            # Indices of the 5 nearest neighbors (excluding the point itself)
            neighbor_indices = indices[i, 1:]
            
            # Calculate the average magnitude of the neighbors
            avg_neighbor_magnitude = np.mean(magnitudes[neighbor_indices])
            
            # If the current point's magnitude is > 150% of the average, mark it as an outlier
            if avg_neighbor_magnitude > 0 and magnitudes[i] > 1.5 * avg_neighbor_magnitude:
                is_outlier[i] = True
        
        num_outliers = np.sum(is_outlier)
        
        # Keep only the non-outlier points
        x_y_diffs = x_y_diffs[~is_outlier]
        image_2_points = image_2_points[~is_outlier]
        
        st.write(f"Removed {num_outliers} outlier(s). {len(x_y_diffs)} vectors remain.")
    else:
        st.write("Skipping outlier removal (not enough data points).")
    # --- END: OUTLIER REMOVAL LOGIC ---

    fig1, ax1 = plt.subplots()
    ax1.plot(x_y_diffs[:, 0], label='x diff')
    ax1.plot(x_y_diffs[:, 1], label='y diff')
    ax1.legend()
    ax1.set_title("Displacement Vectors (x and y) - Outliers Removed")
    
    fig2, ax2 = plt.subplots()
    ax2.imshow(img2, cmap='gray')
    ax2.quiver(image_2_points[:, 0], image_2_points[:, 1], x_y_diffs[:, 0], x_y_diffs[:, 1],
               angles="xy", scale_units='xy', scale=1, color='red', width=0.003)
    ax2.set_title("Displacement Field - Outliers Removed")
    return x_y_diffs, image_2_points, fig1, fig2

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Image Transformation and Warping Analysis App")

def initialize_state():
    defaults = {
        'points': [], 'img1_orig': None, 'img2_orig': None,
        'img1_transformed': None, 'img2_transformed': None,
        'analysis_data': None, 'plot1': None, 'plot2': None,
        'last_coord_1': None, 'last_coord_2': None,
        'meniscus_data': None, 'meniscus_plot': None,
        'theta2_plot': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_state()

with st.sidebar:
    st.header("Controls")
    display_width = st.slider("Image Display Width (pixels)", 200, 1000, 500)
    uploaded_file1 = st.file_uploader("Choose image 1", type=["png", "jpg", "jpeg"])
    if uploaded_file1: st.session_state.img1_orig = Image.open(uploaded_file1)
    uploaded_file2 = st.file_uploader("Choose image 2", type=["png", "jpg", "jpeg"])
    if uploaded_file2: st.session_state.img2_orig = Image.open(uploaded_file2)
    if st.button("Generate Synthetic Speckle Images"):
        ref_img, warped_img = create_and_warp_speckle_image()
        st.session_state.img1_orig = Image.fromarray(ref_img)
        st.session_state.img2_orig = Image.fromarray(warped_img)
        st.session_state.points = []
        st.rerun()

if not st.session_state.img1_orig or not st.session_state.img2_orig:
    st.info("Please upload two images or generate synthetic ones to begin.")
else:
    col1, col2 = st.columns(2)
    new_point = None
    
    w1, h1 = st.session_state.img1_orig.size
    scale1 = display_width / w1
    display_img1 = st.session_state.img1_orig.resize((display_width, int(h1 * scale1)), Image.Resampling.LANCZOS)
    
    w2, h2 = st.session_state.img2_orig.size
    scale2 = display_width / w2
    display_img2 = st.session_state.img2_orig.resize((display_width, int(h2 * scale2)), Image.Resampling.LANCZOS)

    with col1:
        st.header("Image 1 (Click to select)")
        coord1 = streamlit_image_coordinates(display_img1, key="img1_coords")
        if coord1 and coord1 != st.session_state.last_coord_1:
            st.session_state.last_coord_1 = coord1
            new_point = (coord1['x'] / scale1, coord1['y'] / scale1)
    with col2:
        st.header("Image 2 (Click to select)")
        coord2 = streamlit_image_coordinates(display_img2, key="img2_coords")
        if coord2 and coord2 != st.session_state.last_coord_2:
            st.session_state.last_coord_2 = coord2
            new_point = (coord2['x'] / scale2, coord2['y'] / scale2)

    if new_point and len(st.session_state.points) < 2:
        st.session_state.points.append(new_point)
        st.rerun()

    st.header("Transformation Controls")
    st.write("Selected Points (Original Coords):", [(round(p[0]), round(p[1])) for p in st.session_state.points])

    if st.button("Reset Selection"):
        st.session_state.points = []
        st.session_state.last_coord_1 = st.session_state.last_coord_2 = None
        st.rerun()

    if len(st.session_state.points) == 2:
        if st.button("Calculate and Apply Transformation"):
            img1_np, img2_np = np.array(st.session_state.img1_orig.convert('L')), np.array(st.session_state.img2_orig.convert('L'))
            h, w = img1_np.shape[:2]
            trans_matrix = calculate_image_transformation(st.session_state.points, w, h)
            if trans_matrix is not None:
                st.session_state.img1_transformed = cv2.warpAffine(img1_np, trans_matrix, (w, h))
                st.session_state.img2_transformed = cv2.warpAffine(img2_np, trans_matrix, (w, h))
                st.success("Transformation applied successfully!")
            else:
                st.error("Could not estimate transformation. Try selecting different points.")

    if st.session_state.img1_transformed is not None:
        st.header("Transformed Images")
        t_col1, t_col2 = st.columns(2)
        with t_col1: st.image(st.session_state.img1_transformed, caption="Transformed Image 1", width=display_width)
        with t_col2: st.image(st.session_state.img2_transformed, caption="Transformed Image 2", width=display_width)

        st.header("Analysis")
        st.subheader("Physical Parameters")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            length = st.number_input("Real-world length between selected points (mm)", value=20.0, format="%.2f")
        with p_col2:
            water_depth = st.number_input("Depth of water from speckle pattern to surface (mm)", value=10.2, format="%.2f")
        
        if st.button("Analyze Warping and Post-Process"):
            with st.spinner("Analyzing images..."):
                h, w = st.session_state.img1_transformed.shape
                diffs, pts, fig1, fig2 = analyze_warping(st.session_state.img1_transformed, st.session_state.img2_transformed)
                if diffs is not None and len(diffs) > 0:
                    st.session_state.analysis_data = np.hstack((diffs, pts))
                    st.session_state.plot1, st.session_state.plot2 = fig1, fig2
                    
                    initial_df = pl.DataFrame(st.session_state.analysis_data, schema=['x diff', 'y diff', 'init x', 'init y'])
                    final_df, meniscus_fig = diffraction_differences_to_meniscus(initial_df, orig_width=w, depth=water_depth, length=length)
                    
                    st.session_state.meniscus_data = final_df
                    st.session_state.meniscus_plot = meniscus_fig
                    
                    fig3, ax3 = plt.subplots()
                    ax3.imshow(st.session_state.img2_transformed, cmap='gray')
                    
                    df = st.session_state.meniscus_data
                    x, y = df['init x'].to_numpy(), df['init y'].to_numpy()
                    u, v = df['x diff'].to_numpy(), df['y diff'].to_numpy()
                    theta2_mag = df['theta 2 (rad)'].to_numpy()
                    
                    norm = np.sqrt(u**2 + v**2)
                    norm[norm == 0] = 1
                    
                    new_u, new_v = (u / norm) * theta2_mag, (v / norm) * theta2_mag
                    colors = np.abs(theta2_mag)
                    
                    ax3.quiver(x, y, new_u, new_v, colors,
                               angles="xy", scale_units='xy', scale=0.1, cmap='viridis')
                    ax3.set_title("Refraction Angle (Theta 2) Field")
                    st.session_state.theta2_plot = fig3
                    
                    st.success("Analysis complete.")
                else:
                    st.session_state.analysis_data = st.session_state.plot1 = st.session_state.plot2 = None
                    st.session_state.meniscus_data = st.session_state.meniscus_plot = st.session_state.theta2_plot = None
                    
    if st.session_state.meniscus_data is not None:
        st.header("Analysis Results")
        st.dataframe(st.session_state.meniscus_data)
        
        csv_buffer = StringIO()
        st.session_state.meniscus_data.write_csv(csv_buffer)
        
        st.download_button("Download Full Analysis as CSV", csv_buffer.getvalue(), "meniscus_analysis.csv", "text/csv")
        
        st.pyplot(st.session_state.plot1)
        st.pyplot(st.session_state.plot2)
        st.pyplot(st.session_state.theta2_plot)
        st.pyplot(st.session_state.meniscus_plot)