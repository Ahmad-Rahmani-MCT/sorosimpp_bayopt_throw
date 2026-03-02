#!/usr/bin/env python3
import random
import math
import csv
import os

def generate_workspace_targets(num_points=100, inner_radius=0.1, outer_radius=0.35, filename="workspace_targets.csv"):
    targets = []
    
    # Set a seed so you can reproduce these exact same points if needed
    random.seed(42) 

    for _ in range(num_points):
        # Generates a uniform random variable between 0 and 1
        u = random.random()
        
        # Calculate the radius to ensure uniform distribution by area in the annulus
        r = math.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        theta = random.uniform(0, 2 * math.pi)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Rounding to 5 decimal places for clean data
        targets.append([round(x, 5), round(y, 5)])

    # Get the directory of the current script to save the CSV alongside it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Write the points to a CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target_x', 'target_y']) # Header row
        writer.writerows(targets)

    print(f"--- SUCCESS ---")
    print(f"Generated {num_points} uniformly distributed points in an annulus.")
    print(f"Inner Radius: {inner_radius} | Outer Radius: {outer_radius}")
    print(f"Saved to: {file_path}")

if __name__ == '__main__':
    generate_workspace_targets()