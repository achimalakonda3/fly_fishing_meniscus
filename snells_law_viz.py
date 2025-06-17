import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

angles = np.linspace(-1, -45, 20)
print(angles)

os.makedirs("frames", exist_ok=True)
frame_paths = []

for interface_tilt_angle in angles:
    # Drawing the water-air interface
    interface_slope = np.tan(np.deg2rad(interface_tilt_angle))
    rotated_x_axis_points = np.arange(-5,5,0.1)
    rotated_y_axis_points = interface_slope * rotated_x_axis_points

    # Drawing the transmitted light ray coming out of the water-air interface
    transmitted_light_ray_angle = 90
    transmitted_ray_x_points = np.array([0,0])
    transmitted_ray_y_points = np.array([0,5])

    # Drawing the incident light coming towards the water-air interface
    theta_2 = 90 - interface_tilt_angle
    # print(np.rad2deg(theta_2))
    theta_1 = np.arcsin(np.sin(np.deg2rad(theta_2))/1.333)
    # print(np.rad2deg(theta_1))
    incident_light_ray_angle = interface_tilt_angle +90+ np.rad2deg(theta_1)
    # print(incident_light_ray_angle)
    incident_light_ray_slope = np.tan(np.deg2rad(incident_light_ray_angle))
    # if incident_light_ray_angle > 100:
    incident_light_ray_x_points = np.arange(-0.1,5,0.1)
    # else:
        # incident_light_ray_x_points = np.arange(-5,0.1,0.1)
    incident_light_ray_y_points = incident_light_ray_x_points * incident_light_ray_slope

    fig, ax = plt.subplots()
    ax.plot(rotated_x_axis_points, rotated_y_axis_points, 'b')
    ax.plot(rotated_y_axis_points, -rotated_x_axis_points, 'b')

    ax.plot(transmitted_ray_x_points, transmitted_ray_y_points, 'r')
    ax.plot(incident_light_ray_x_points, incident_light_ray_y_points)
    ax.set_ylim(-5, 5)
    ax.set_aspect(1)
    ax.axis('off')

    # save to buffer
    filename = f"frame_{interface_tilt_angle:.1f}.png"
    print(f"saved {filename}")
    plt.savefig(filename)
    frame_paths.append(filename)
    plt.close()

# Create the forward + backward frame sequence
all_frames = frame_paths + frame_paths[-2:0:-1]  # Avoid repeating first/last

# Create GIF
images = [imageio.imread(f) for f in all_frames]
imageio.mimsave("snells_law_loop.gif", images, duration=0.1)

# Optional: clean up
for f in frame_paths:
    os.remove(f)