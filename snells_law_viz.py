import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

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

if __name__ == "__main__":

    interface_angles = np.linspace(0, -90, 40)
    theta_2s = -np.deg2rad(interface_angles)
    n1 = 1.333
    n2 = 1.0
    theta_1s = (np.arcsin(n2* np.sin(theta_2s) / n1) + (np.pi +theta_2s))
    horizs = np.pi/2 - theta_1s - interface_angles
    plt.plot( -interface_angles, np.rad2deg(theta_1s)-180)
    plt.title("Interface Angle vs Incident Ray Angle")
    plt.xlabel("Interface Angle (degrees)")
    plt.ylabel("Incident Ray Angle (degrees)")
    plt.show()