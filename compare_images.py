import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import numpy as np
import cv2
import polars as pl
import plotly.express as px
from io import StringIO
from streamlit_image_coordinates import streamlit_image_coordinates
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from scipy.optimize import fsolve
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid

# This function performs calculations and returns the final DataFrame and a plot figure
def diffraction_differences_to_meniscus(diffraction_data_df, orig_width, depth = 14/32 * 23.5, water_RI = 1.3333, length = 20):
    """
    Takes a Polars DataFrame with raw displacement data, calculates meniscus
    angles based on Snell's Law, and returns an enriched DataFrame and a plot.
    """
    pixels_for_length = orig_width * 0.3
    mm_per_px = length / pixels_for_length
    
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

    diffraction_data_filtered = diffraction_data_df.filter(pl.col("init x").is_between(700, 900))

    fig, ax = plt.subplots()
    ax.plot(diffraction_data_filtered["init y (mm)"], diffraction_data_filtered["theta 2 (deg)"], '.b')
    ax.set_xlabel("Y Position (mm)")
    ax.set_ylabel("Water Surface Angle (degrees)")
    ax.set_title("Y Position vs Water Surface Angle")
    
    return diffraction_data_df, diffraction_data_filtered, fig

def interpolate_and_plot_theta2_grid(meniscus_data, width, height):
    """
    Interpolates scattered theta 2 data onto a full pixel grid and plots it as a heatmap.
    """
    st.write("Interpolating theta 2 field...")
    points = meniscus_data[['init x', 'init y']].to_numpy()
    values = meniscus_data['theta 2 (deg)'].to_numpy()
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    interpolated_grid = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
    fig, ax = plt.subplots()
    im = ax.imshow(interpolated_grid, cmap='viridis', origin='lower', aspect='auto')
    cbar = fig.colorbar(im)
    cbar.set_label("Refraction Angle (Theta 2) in Degrees")
    ax.set_title("Interpolated Theta 2 Field (Heatmap)")
    ax.set_xlabel("X pixel coordinate")
    ax.set_ylabel("Y pixel coordinate")
    return fig

def reconstruct_and_plot_surface(meniscus_data, width, height, mm_per_px, max_dist_px=200):
    # --- Function logic is identical up to the point of creating the final dataframe ---
    st.write("Reconstructing 3D surface from integrated slopes...")
    points = meniscus_data[['init x', 'init y']].to_numpy()
    meniscus_data_with_slopes = meniscus_data.with_columns(
    [
        (
            (pl.col('theta 2 (rad)').tan()) *
            (pl.col('x diff') / ((pl.col('x diff')**2 + pl.col('y diff')**2).sqrt()))
        ).alias('slope_x_component'),
        (
            (pl.col('theta 2 (rad)').tan()) *
            (pl.col('y diff') / ((pl.col('x diff')**2 + pl.col('y diff')**2).sqrt()))
        ).alias('slope_y_component')
    ]
    )
    slopes_x = meniscus_data_with_slopes['slope_x_component'].to_numpy()
    slopes_y = meniscus_data_with_slopes['slope_y_component'].to_numpy()
    if points.shape[0] == 0:
        st.warning("No data points found to build surface.")
        # Return Nones for all expected outputs
        return None, None, None, None, None, None, None
    kdtree = KDTree(points)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    distances, _ = kdtree.query(grid_points, k=1)
    distance_grid = distances.reshape(grid_x.shape)
    interpolated_slopes_x = griddata(points, slopes_x, (grid_x, grid_y), method='linear', fill_value=0)
    interpolated_slopes_y = griddata(points, slopes_y, (grid_x, grid_y), method='linear', fill_value=0)
    height_grid_px_y_pos = -cumulative_trapezoid(interpolated_slopes_y, axis=0, initial=0)
    height_grid_px_y_neg = cumulative_trapezoid(interpolated_slopes_y[::-1, :], axis=0, initial=0)[::-1, :]
    height_grid_px_x_pos = cumulative_trapezoid(interpolated_slopes_x, axis=1,initial=0)
    height_grid_px_x_neg = cumulative_trapezoid(interpolated_slopes_x[:, ::-1], axis=1,initial=0)[:, ::-1]
    n,m = np.shape(height_grid_px_x_pos)
    vertical_multiplier_pos = np.tile(np.linspace(1, 0, n), (m,1)).transpose()
    vertical_multiplier_neg =  np.tile(np.linspace(0, 1, n), (m,1)).transpose()
    horizontal_multiplier_pos = np.tile(np.linspace(1, 0, m), (n,1))
    horizontal_multiplier_neg = np.tile(np.linspace(0, 1, m), (n,1))
    valid_data_mask = ~(distance_grid > max_dist_px)
    uint8_mask = valid_data_mask.astype(np.uint8)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(uint8_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels_im != largest_label)
    else:
        mask = valid_data_mask
    vertical_multiplier_pos[mask] = 0
    vertical_multiplier_neg[mask] = 0
    horizontal_multiplier_pos[mask] = 0
    horizontal_multiplier_neg[mask] = 0
    multiplier_fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    multiplier_fig.suptitle('Multiplier Matrix Heatmaps', fontsize=16)
    im1 = axs[0, 0].imshow(vertical_multiplier_pos, aspect='auto', cmap='viridis')
    axs[0, 0].set_title("Vertical Multiplier (Positive)")
    multiplier_fig.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].imshow(vertical_multiplier_neg, aspect='auto', cmap='viridis')
    axs[0, 1].set_title("Vertical Multiplier (Negative)")
    multiplier_fig.colorbar(im2, ax=axs[0, 1])
    im3 = axs[1, 0].imshow(horizontal_multiplier_pos, aspect='auto', cmap='viridis')
    axs[1, 0].set_title("Horizontal Multiplier (Positive)")
    multiplier_fig.colorbar(im3, ax=axs[1, 0])
    im4 = axs[1, 1].imshow(horizontal_multiplier_neg, aspect='auto', cmap='viridis')
    axs[1, 1].set_title("Horizontal Multiplier (Negative)")
    multiplier_fig.colorbar(im4, ax=axs[1, 1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    total_weight = (vertical_multiplier_pos + vertical_multiplier_neg + horizontal_multiplier_pos + horizontal_multiplier_neg)
    eps = 1e-12
    total_weight_safe = total_weight + eps
    height_grid_px = ((height_grid_px_y_pos * vertical_multiplier_pos
                       + height_grid_px_y_neg * vertical_multiplier_neg
                       + height_grid_px_x_pos * horizontal_multiplier_pos
                       + height_grid_px_x_neg * horizontal_multiplier_neg)
                    / total_weight_safe)
    
    # --- Keep a copy of the raw height data before masking for volume calculation ---
    raw_height_grid_for_volume = np.copy(height_grid_px)

    border_vals = np.concatenate([height_grid_px[0, :], height_grid_px[-1, :], height_grid_px[:, 0], height_grid_px[:, -1]])
    offset = np.mean(border_vals)
    height_grid_px -= offset
    height_grid_px *= -1
    st.write(f"Masking out {np.sum(mask)} points further than {max_dist_px} pixels or outside the main data area.")
    height_grid_px[mask] = np.nan
    x_mm, y_mm, z_mm = grid_x * mm_per_px, grid_y * mm_per_px, 50*height_grid_px * mm_per_px
    point_cloud_df = pl.DataFrame({"x": x_mm.ravel(), "y": y_mm.ravel(), "z": z_mm.ravel()})
    point_cloud_df = point_cloud_df.filter(pl.col("z").is_not_nan())
    max_points_to_plot = 50000
    if point_cloud_df.height > max_points_to_plot:
        st.write(f"Downsampling from {point_cloud_df.height} to {max_points_to_plot} random points for plotting.")
        point_cloud_df = point_cloud_df.sample(n=max_points_to_plot, seed=1)
    st.write(f"Plotting {point_cloud_df.height} points.")
    if point_cloud_df.height == 0:
        st.warning(f"No data points left to plot after filtering with max_dist_px={max_dist_px}. Try increasing this value.")
        return None, None, None, None, None, None, None
        
    # --- Plotting logic remains the same ---
    fig = px.scatter_3d(point_cloud_df.to_pandas(), x="x", y="y", z="z", color="z", color_continuous_scale='Viridis')
    fig.update_layout(scene_aspectmode='data', coloraxis_colorbar=dict(title="Z-Value (mm * 50)"))
    fig.update_traces(marker=dict(size=2, line=dict(width=1, color='DarkSlateGray')))

    # --- MODIFIED RETURN STATEMENT ---
    return fig, point_cloud_df, multiplier_fig, raw_height_grid_for_volume, mask, x_mm, y_mm

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
    target_half_width = (orig_width * 0.3) / 2
    dst_pts_orig = np.float32([
        [(orig_width / 2) - target_half_width, target_center_y],
        [(orig_width / 2) + target_half_width, target_center_y]
    ])
    transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts_orig, dst_pts_orig)
    return transformation_matrix

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

    if len(x_y_diffs) > 5:
        st.write("Performing outlier removal based on local vector magnitudes...")
        magnitudes = np.linalg.norm(x_y_diffs, axis=1)
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(image_2_points)
        distances, indices = nbrs.kneighbors(image_2_points)
        is_outlier = np.zeros(len(magnitudes), dtype=bool)
        for i in range(len(magnitudes)):
            neighbor_indices = indices[i, 1:]
            avg_neighbor_magnitude = np.mean(magnitudes[neighbor_indices])
            if avg_neighbor_magnitude > 0 and magnitudes[i] > 1.5 * avg_neighbor_magnitude:
                is_outlier[i] = True
        num_outliers = np.sum(is_outlier)
        x_y_diffs = x_y_diffs[~is_outlier]
        image_2_points = image_2_points[~is_outlier]
        st.write(f"Removed {num_outliers} outlier(s). {len(x_y_diffs)} vectors remain.")
    else:
        st.write("Skipping outlier removal (not enough data points).")

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

def calculate_and_plot_volume(height_grid_px, mask, x_mm, y_mm, mm_per_px, border_px=40):
    """
    Calculates the volume under a reconstructed surface and creates a new plot
    with colors based on whether points are above or below a calculated zero-plane.
    """
    st.write("---")
    st.header("Volume Analysis")

    # --- 1. Define the zero-plane based on a border region ---
    st.write(f"Calculating zero-plane from a {border_px}px border...")
    n, m = height_grid_px.shape
    
    # Create a mask that is True only for the border region
    border_mask = np.zeros_like(height_grid_px, dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True
    
    # Combine the border mask with the valid data mask to get the reference points
    reference_points_mask = border_mask & ~mask
    
    # Get the height values of these reference points
    border_heights = height_grid_px[reference_points_mask]
    
    if len(border_heights) == 0:
        st.warning("No valid data points found in the border region to calculate a zero-plane. Using mean of all points.")
        border_heights = height_grid_px[~mask]
        if len(border_heights) == 0:
            st.error("No valid data points available for volume calculation.")
            return 0, 0, None

    zero_level = np.mean(border_heights)
    st.write(f"Calculated zero-level offset: {zero_level:.4f} (in pixel height units)")
    
    # Adjust the entire height grid relative to this new zero-plane
    adjusted_height_grid = height_grid_px - zero_level
    
    # --- 2. Calculate volumes ---
    # Apply the mask to the adjusted grid, setting invalid areas to zero for calculation
    adjusted_height_grid[mask] = 0
    
    # Calculate the area of a single grid cell in mm^2
    cell_area = mm_per_px * mm_per_px
    
    # Separate positive and negative heights
    positive_heights = np.copy(adjusted_height_grid)
    positive_heights[positive_heights < 0] = 0
    
    negative_heights = np.copy(adjusted_height_grid)
    negative_heights[negative_heights > 0] = 0
    
    # Volume is the sum of heights multiplied by cell area and the 50x z-scaling factor
    z_scale_factor = 1
    volume_above_zero = np.sum(positive_heights) * cell_area * mm_per_px * z_scale_factor
    volume_below_zero = np.sum(negative_heights) * cell_area * mm_per_px * z_scale_factor

    # --- 3. Create the new color-coded plot ---
    # Prepare data for plotting
    adjusted_height_grid[mask] = np.nan # Re-apply mask as NaN for plotting
    z_mm_adjusted = adjusted_height_grid * mm_per_px * z_scale_factor

    # Create a categorical column for color
    level = np.full(z_mm_adjusted.shape, "At Zero-Plane", dtype=object)
    level[z_mm_adjusted > 0] = "Above Zero-Plane"
    level[z_mm_adjusted < 0] = "Below Zero-Plane"
    
    # Build the dataframe for plotting
    volume_plot_df = pl.DataFrame({
        "x": x_mm.ravel(),
        "y": y_mm.ravel(),
        "z": z_mm_adjusted.ravel(),
        "level": level.ravel()
    }).filter(pl.col("z").is_not_nan())

    st.write(f"Plotting {volume_plot_df.height} points for volume visualization.")
    if volume_plot_df.height == 0:
        return volume_above_zero, volume_below_zero, None
        
    fig = px.scatter_3d(
        volume_plot_df.to_pandas(),
        x='x', y='y', z='z',
        color='level',
        color_discrete_map={
            'Above Zero-Plane': 'yellow',
            'Below Zero-Plane': 'darkblue'
        },
        category_orders={"level": ["Below Zero-Plane", "At Zero-Plane", "Above Zero-Plane"]} # Ensures legend order
    )
    fig.update_layout(
        title="Surface Relative to Border Zero-Plane",
        scene_aspectmode='data',
        legend_title="Position"
    )
    fig.update_traces(marker=dict(size=2))

    return volume_above_zero, volume_below_zero, fig


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
        'theta2_plot': None, 'theta2_grid_plot': None,
        'pydeck_chart': None,
        'reconstructed_surface_data': None,
        'multiplier_plot': None, # State for the new multiplier plot
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
                    final_df, filtered_df, meniscus_fig = diffraction_differences_to_meniscus(initial_df, orig_width=w, depth=water_depth, length=length)
                    
                    st.session_state.meniscus_data = final_df
                    st.session_state.filtered_data = filtered_df
                    st.session_state.meniscus_plot = meniscus_fig
                    
                    pixels_for_length = w * 0.3
                    mm_per_px = length / pixels_for_length

                    # --- UPDATED FUNCTION CALL ---
                    # Unpack all the new return values
                    _3d_figure, point_cloud_df, multiplier_fig, raw_h_grid, final_mask, x_grid, y_grid = reconstruct_and_plot_surface(
                        st.session_state.meniscus_data, w, h, mm_per_px
                    )
                    
                    # Check if the reconstruction was successful before proceeding
                    if _3d_figure:
                        st.session_state._3d_figure = _3d_figure
                        st.session_state.point_cloud_df = point_cloud_df
                        st.session_state.multiplier_plot = multiplier_fig 

                        # --- CALL THE NEW VOLUME FUNCTION ---
                        vol_pos, vol_neg, vol_fig = calculate_and_plot_volume(
                            raw_h_grid, final_mask, x_grid, y_grid, mm_per_px
                        )
                        st.session_state.volume_positive = vol_pos
                        st.session_state.volume_negative = vol_neg
                        st.session_state.volume_plot = vol_fig
                    
                    st.success("Analysis complete.")
                else:
                    st.session_state.analysis_data = st.session_state.plot1 = st.session_state.plot2 = None
                    st.session_state.meniscus_data = st.session_state.meniscus_plot = st.session_state.theta2_plot = st.session_state.theta2_grid_plot = st.session_state.pydeck_chart = st.session_state.reconstructed_surface_data = st.session_state.multiplier_plot = None
                    
    if st.session_state.meniscus_data is not None:
        st.header("Analysis Results")
        st.dataframe(st.session_state.meniscus_data)
        
        # ... (rest of your existing plotting code for plot1, plot2, etc.) ...
        st.pyplot(st.session_state.plot2)
        st.pyplot(st.session_state.meniscus_plot)
        st.dataframe(st.session_state.filtered_data)

        if st.session_state.multiplier_plot:
            st.header("Multiplier Matrix Visualization")
            st.pyplot(st.session_state.multiplier_plot)
        
        st.header("Reconstructed 3D Surface")
        # Check if the figure exists before trying to plot it
        if st.session_state._3d_figure:
            st.plotly_chart(st.session_state._3d_figure, use_container_width=True)
            st.dataframe(st.session_state.point_cloud_df)
            # ... (your download button logic) ...

        # --- NEW SECTION TO DISPLAY VOLUME RESULTS ---
        if 'volume_plot' in st.session_state and st.session_state.volume_plot:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.metric(label="Volume Above Zero-Plane (mm³)", value=f"{st.session_state.volume_positive:,.2f}")
            with v_col2:
                st.metric(label="Volume Below Zero-Plane (mm³)", value=f"{st.session_state.volume_negative:,.2f}")

            st.plotly_chart(st.session_state.volume_plot, use_container_width=True)