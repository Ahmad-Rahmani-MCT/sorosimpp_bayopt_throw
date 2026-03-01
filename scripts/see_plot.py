#!/usr/bin/env python3
import csv
import os
import matplotlib.pyplot as plt

def plot_workspace_targets(filename="workspace_targets.csv"):
    # Get the directory of the current script to locate the CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    x_coords = []
    y_coords = []

    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"Error: Could not find '{file_path}'. Did you run the generator script first?")
        return

    # Read the data from the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader) # Skip the header row ('target_x', 'target_y')
        for row in reader:
            x_coords.append(float(row[0]))
            y_coords.append(float(row[1]))

    # Set up the plot area
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot the loaded coordinates
    ax.scatter(x_coords, y_coords, color='blue', s=15, alpha=0.7, label='Targets')

    # Draw the boundary circles for visual reference
    inner_circle = plt.Circle((0, 0), 0.1, color='red', fill=False, linestyle='--', linewidth=1.5, label='Inner Boundary (0.1)')
    outer_circle = plt.Circle((0, 0), 0.25, color='green', fill=False, linestyle='--', linewidth=1.5, label='Outer Boundary (0.25)')
    
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    # CRITICAL: Set aspect ratio to 'equal' so the circles don't look like stretched ovals
    ax.set_aspect('equal', adjustable='box')
    
    # Give the plot some breathing room around the edges
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])

    # Labels, title, and grid formatting
    ax.set_xlabel('Target X')
    ax.set_ylabel('Target Y')
    ax.set_title('Workspace Target Distribution')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

    print(f"Loaded {len(x_coords)} points from {filename}.")
    print("Displaying plot. Close the plot window to finish.")
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    plot_workspace_targets()