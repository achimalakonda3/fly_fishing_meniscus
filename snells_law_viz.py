import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import imageio
import os
from tkinter import filedialog
from tkinter import Tk
import polars as pl
from scipy.optimize import fsolve
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def gif_generator():
    angles = np.linspace(0, -90, 40)
    angles = np.deg2rad(angles)


    os.makedirs("frames", exist_ok=True)
    frame_paths = []
    theta_1s = []
    incident_ray_wrt_horizs = []

    for interface_angle in angles:
        
        # Plot Interface Lines
        fig, ax = plt.subplots()
        ax.arrow(0,0, np.cos(interface_angle), np.sin(interface_angle), width = 0.05, head_width = 0) # Interface Line
        ax.arrow(0,0, -np.cos(interface_angle), -np.sin(interface_angle), width = 0.05, head_width = 0) # Interface Line

        # Plot Perpencidular Interface Lines
        ax.arrow(0,0, np.sin(interface_angle), -np.cos(interface_angle), ls=':')
        ax.arrow(0,0, -np.sin(interface_angle), np.cos(interface_angle), ls=':')

        # Calculate with Snell's Law
        n1 = 1.333
        n2 = 1.0
        # theta_1 = np.pi/12
        # theta_2 = np.arcsin(n1*np.sin(theta_1) / n2)
        theta_2 = -interface_angle
        theta_1 = np.arcsin(n2* np.sin(theta_2) / n1) + (np.pi - interface_angle)
        theta_1s.append(theta_1)

        # Plot light rays
        incident_ray_wrt_horiz = np.pi/2 - theta_1 - interface_angle
        incident_ray_length = 1
        incident_ray_wrt_horizs.append(incident_ray_wrt_horiz)

        transmitted_ray_wrt_horiz = np.pi/2 - theta_2 - interface_angle
        incident_ray_length = 1

        ax.arrow(np.cos(incident_ray_wrt_horiz), np.sin(incident_ray_wrt_horiz), 
                -np.cos(incident_ray_wrt_horiz), -np.sin(incident_ray_wrt_horiz), 
                width = 0.05,
                length_includes_head = True, 
                color = 'b') # Incident Ray
        ax.arrow(0, 0, 
                np.cos(transmitted_ray_wrt_horiz), np.sin(transmitted_ray_wrt_horiz), 
                width = 0.05,
                length_includes_head = True,
                color = 'r') # Transmitted Ray
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_aspect(1)
        ax.axis('off')
        # ax.legend(['Interface', 'Incident Light Ray', 'Transmitted Light Ray'], loc = 'lower right')
        # plt.show()

        # save to buffer
        filename = f"frames/frame_{np.rad2deg(interface_angle):.1f}.png"
        print(f"saved {filename}")
        plt.savefig(filename)
        frame_paths.append(filename)
        plt.close()
    
    # Create the forward + backward frame sequence
    all_frames = frame_paths + frame_paths[-2:0:-1]  # Avoid repeating first/last

    # Create GIF
    images = [imageio.imread(f) for f in all_frames]
    imageio.mimsave("snells_law_loop_0_to_90_deg.gif", images, duration=0.1)

    # Optional: clean up
    # for f in frame_paths:
    #     os.remove(f)

def diffraction_differences_to_meniscus(parent_root = None):
    if parent_root is None:
        root = Tk()
        root.withdraw
    else:
        root = parent_root
    path = filedialog.askopenfilename( parent=root,
                title=f"Select csv file",
                filetypes=[("csv files", "*.csv")]
            )
    if not path:
        print("No file selected. Exiting function.")
        # If a temporary root was created, destroy it.
        if parent_root is None:
            root.destroy()
        return
    
    diffraction_data = pl.read_csv(path)
    mm_per_px = 20 / (1280 - 320)
    depth = 11.00 # mm
    norm_mm_diff = ( mm_per_px * (pl.col("y diff")**2 + pl.col("x diff")**2).sqrt() )
    diffraction_data = diffraction_data.with_columns(pl.arctan2(norm_mm_diff, pl.lit(depth)).alias("beta"))
    init_x_mm = pl.col("init x") * mm_per_px
    diffraction_data = diffraction_data.with_columns((init_x_mm).alias("init x (mm)"))
    init_y_mm = pl.col("init y") * mm_per_px
    diffraction_data = diffraction_data.with_columns((init_y_mm).alias("init y (mm)"))

    betas = diffraction_data["beta"].to_numpy()

    def snells_law_eqns(thetas, beta):
        return [thetas[0] + beta - thetas[1], 
                np.sin(thetas[0]) - 1.3333*np.sin(thetas[1])]
    
    theta_1s = []
    theta_2s = []

    # Get the angle of the surface with respect to the horizontal
    for b in betas:
        soln = fsolve(snells_law_eqns, x0=[0,0], args=(b,))
        theta_1s.append(soln[0])
        theta_2s.append(soln[1])
    
    diffraction_data = diffraction_data.with_columns([
        pl.Series("theta 1", theta_1s),
        pl.Series("theta 2", theta_2s),
        pl.Series("theta 1 (deg)", np.degrees(theta_1s)),
        pl.Series("theta 2 (deg)", np.degrees(theta_2s)),
    ])

    plt.plot(diffraction_data["init y (mm)"], diffraction_data["theta 1 (deg)"], '.b')
    plt.xlabel("Y Position (mm)")
    plt.ylabel("Water Surface Angle (degrees)")
    plt.title("Y Position vs Water Surface Angle")
    plt.show()
    diffraction_data.write_csv(path)
    

def clear_edge_points_and_plot(parent_root = None):
    if parent_root is None:
        root = Tk()
        root.withdraw
    else:
        root = parent_root
    path = filedialog.askopenfilename( parent=root,
                title=f"Select csv file",
                filetypes=[("csv files", "*.csv")]
            )
    if not path:
        print("No file selected. Exiting function.")
        # If a temporary root was created, destroy it.
        if parent_root is None:
            root.destroy()
        return

    diffraction_data = pl.read_csv(path)

    diffraction_data = diffraction_data.filter(pl.col("init x").is_between(400, 1200))
    plt.plot(diffraction_data["init y (mm)"], diffraction_data["theta 1 (deg)"], '.b')
    plt.xlabel("Y Position (mm)")
    plt.ylabel("Water Surface Angle (degrees)")
    plt.title("Y Position vs Water Surface Angle")
    plt.show()

def surface_integral_calcs(parent_root = None):
    if parent_root is None:
        root = Tk()
        root.withdraw
    else:
        root = parent_root
    path = filedialog.askopenfilename( parent=root,
                title=f"Select csv file",
                filetypes=[("csv files", "*.csv")]
            )
    if not path:
        print("No file selected. Exiting function.")
        # If a temporary root was created, destroy it.
        if parent_root is None:
            root.destroy()
        return
    
    diffraction_data = pl.read_csv(path)
    diffraction_data = diffraction_data.with_columns(pl.col("theta 2").tan().abs().alias("gradient magnitude"))
    gradient_x_calc = pl.col("gradient magnitude") * pl.col("x diff") / (pl.col("x diff")**2 + pl.col("y diff")**2).sqrt()
    gradient_y_calc = pl.col("gradient magnitude") * pl.col("y diff") / (pl.col("x diff")**2 + pl.col("y diff")**2).sqrt()
    diffraction_data = diffraction_data.with_columns(gradient_x_calc.alias("gradient x"))
    diffraction_data = diffraction_data.with_columns(gradient_y_calc.alias("gradient y"))
    # diffraction_data.write_csv(path)

    gx = diffraction_data["gradient x"].to_numpy()
    gy = diffraction_data["gradient y"].to_numpy()
    
    init_x_numpy_array = diffraction_data["init x (mm)"].to_numpy()
    init_y_numpy_array = diffraction_data["init y (mm)"].to_numpy()
    points = np.vstack((init_x_numpy_array, init_y_numpy_array)).transpose()

    matrix = np.column_stack((np.ones(points.shape[0]), points))    
    print(matrix)

    tri = Delaunay(points)
    simplices = tri.simplices
    print("Triangulation:", simplices.shape[0], "triangles,", points.shape[0], "points")

    # Prepare global FEM assembly
    Nnodes = points.shape[0]
    K = lil_matrix((Nnodes, Nnodes), dtype=np.float64)
    b = np.zeros(Nnodes, dtype=np.float64)

    # For each triangle: fit planes to gx and gy (linear -> constant derivatives)
    for simplex in simplices:
        inds = simplex
        pts = points[inds]          # shape (3,2)
        gx_vals = gx[inds]
        gy_vals = gy[inds]

        # Build matrix [1 x y] for the three vertices
        M = np.column_stack((np.ones(3), pts))  # shape (3,3)
        # compute area (signed)
        area = 0.5 * np.linalg.det(M)
        if area <= 0:
            # degenerate triangle? skip
            continue
        # Solve for plane coefficients: v ~ a + b*x + c*y
        coeffs_gx = np.linalg.solve(M, gx_vals)   # [a, b, c]
        coeffs_gy = np.linalg.solve(M, gy_vals)

        # partial derivatives inside triangle (constant)
        dgx_dx = coeffs_gx[1]
        dgx_dy = coeffs_gx[2]
        dgy_dx = coeffs_gy[1]
        dgy_dy = coeffs_gy[2]

        # divergence (source term f) in this triangle
        f_tri = dgx_dx + dgy_dy

        # Element stiffness matrix for linear triangle
        # Compute gradients of barycentric (shape) functions:
        # inverse of M: invM @ [1,0,0]^T gives coefficients of phi1 etc.
        invM = np.linalg.inv(M)       # shape (3,3)
        grads = invM[1:, :]           # shape (2,3): rows are d/dx and d/dy for each shape fn
        # Ke = area * (grads.T @ grads)
        Ke = area * (grads.T @ grads)  # (3,3)

        # Element load vector: integrate f * phi_i over triangle => f_tri * area / 3 for each node (linear)
        be = f_tri * area / 3.0 * np.ones(3)

        # Assemble
        for a in range(3):
            A = inds[a]
            b[A] += be[a]
            for c in range(3):
                C = inds[c]
                K[A, C] += Ke[a, c]

    # Apply Dirichlet BC: set one reference node to zero (remove one DOF)
    # Choose node with smallest x (arbitrary stable choice)
    ref_node = np.argmin(points[:,0])
    print("Reference node (z=0):", ref_node, "at", points[ref_node])

    all_nodes = np.arange(Nnodes)
    free_nodes = np.setdiff1d(all_nodes, [ref_node])

    K_free = K[free_nodes, :][:, free_nodes].tocsr()
    b_free = b[free_nodes] - K[free_nodes, :][:, [ref_node]].toarray().flatten() * 0.0  # z_ref = 0, so no contribution

    print("Solving linear system with", K_free.shape[0], "unknowns...")
    z_free = spsolve(K_free, b_free)

    z = np.zeros(Nnodes, dtype=float)
    z[free_nodes] = z_free
    z[ref_node] = 0.0

    # Optional: shift so min z = 0 for nicer plotting
    z = z - np.min(z)

    # Plot using triangulation
    triang = mtri.Triangulation(points[:,0], points[:,1], simplices)
    plt.figure(figsize=(8,6))
    tcf = plt.tricontourf(triang, z, levels=60, cmap="terrain")
    plt.colorbar(tcf, label="Elevation (arb. units)")
    plt.triplot(triang, color='k', linewidth=0.3, alpha=0.3)
    plt.scatter(points[ref_node,0], points[ref_node,1], color='r', s=20, label='reference (z=0)')
    plt.legend()
    plt.title("Recovered Elevation from Gradient Field (FEM Poisson)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()




    
    
    


if __name__ == "__main__":
    # interface_angles = np.linspace(0, -90, 40)
    # theta_2s = -np.deg2rad(interface_angles)
    # n1 = 1.333
    # n2 = 1.0
    # theta_1s = (np.arcsin(n2* np.sin(theta_2s) / n1) + (np.pi +theta_2s))
    # horizs = np.pi/2 - theta_1s - interface_angles
    # plt.plot( -interface_angles, np.rad2deg(theta_1s)-180)
    # plt.title("Interface Angle vs Incident Ray Angle")
    # plt.xlabel("Interface Angle (degrees)")
    # plt.ylabel("Incident Ray Angle (degrees)")
    # plt.show()

    surface_integral_calcs()