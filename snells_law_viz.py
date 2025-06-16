import matplotlib.pyplot as plt
import numpy as np


# Drawing the water-air interface
interface_tilt_angle = -30
interface_slope = np.tan(np.deg2rad(interface_tilt_angle))
rotated_x_axis_points = np.arange(-5,5,0.1)
rotated_y_axis_points = interface_slope * rotated_x_axis_points

# Drawing the transmitted light ray coming out of the water-air interface
transmitted_light_ray_angle = 90
transmitted_ray_x_points = np.array([0,0])
transmitted_ray_y_points = np.array([0,5])

# Drawing the incident light coming towards the water-air interface
theta_2 = 90 - interface_tilt_angle
print(np.rad2deg(theta_2))
theta_1 = np.arcsin(np.sin(np.deg2rad(theta_2))/1.333)
print(np.rad2deg(theta_1))
incident_light_ray_angle = interface_tilt_angle + np.rad2deg(theta_1)
print(incident_light_ray_angle)
incident_light_ray_slope = np.tan(np.deg2rad(incident_light_ray_angle))
incident_light_ray_x_points = np.arange(-5,0.1,0.1)
incident_light_ray_y_points = incident_light_ray_x_points * incident_light_ray_slope

fig, ax = plt.subplots()
ax.plot(rotated_x_axis_points, rotated_y_axis_points, 'b')
ax.plot(rotated_y_axis_points, -rotated_x_axis_points, 'b')

ax.plot(transmitted_ray_x_points, transmitted_ray_y_points, 'r')
ax.plot(incident_light_ray_x_points, incident_light_ray_y_points)
ax.set_ylim(-5, 5)
ax.set_aspect(1)

plt.show()
