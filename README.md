# ENPM661 - Project 03 - Phase 1  
**Implementation of the A\* Algorithm for a Mobile Robot**

## Overview

This project implements the A\* search algorithm for a mobile robot that navigates a 2D map. The robot (with a radius of 5 mm and a clearance of 5 mm) must find an optimal path from a user-defined start to a goal coordinate while avoiding obstacles. The project uses the mathematical representation of free space (using half-planes and semi-algebraic models) and visualizes both the node exploration process and the final optimal path.

## Project Structure

- **MapEnv Class**  
  - Constructs the map by scaling raw coordinates to a 600×250 canvas. These coordinates have been taken from the map provided in Project 2 of the course - ENPM661 - Planning for Autonomous Robots at The University Of Maryland, College Park.
  - Inflates obstacles by the sum of clearance and robot radius.
  - Creates both polygon obstacles and circular/arc obstacles.
  - Provides methods to draw the obstacles and clearances, and to check for collisions (via half-planes).

- **aStar Class**  
  - Implements the A\* search algorithm.
  - Defines five motion primitives (`forward`, `forwardLeft30`, `forwardLeft60`, `forwardRight30`, `forwardRight60`) using a helper function that leverages precomputed trigonometric values.
  - Uses a priority queue (via the PriorityQueue class) to expand nodes and backtracks to reconstruct the optimal path.
  - Incorporates dynamic discretization (with thresholds of 0.5 for x,y and 30° for theta) to prevent duplicate node exploration.

- **Node Class**  
  - Represents a state in the search space.
  - Stores the current state (position and orientation), cost-to-come, heuristic cost-to-go, and a pointer to the parent node.
  - Supports comparison and hashing for use in the A* search algorithm.

- **PriorityQueue Class**  
  - Manages the open set using Python’s `heapq` and dictionaries for efficient lookup and updating of nodes.

- **Visualizer Class**  
  - Handles drawing of start and goal points, obstacles, and clearances.
  - Animates node exploration and the final path using OpenCV.
  - For static displays, matplotlib is used to show images with labeled x–y axes.

- **Inputs Class**  
  - Manages user input for start and goal coordinates as well as the robot’s step size.
  - Includes collision checking (using MapEnv’s halfPlanes method) to ensure that start and goal positions are in free space.

## How to Run

1. **Dependencies:**  
   - Python 3.x  
   - [NumPy](https://numpy.org)  
   - [OpenCV](https://opencv.org/)  
   - [Matplotlib](https://matplotlib.org/)  
   - [tqdm](https://tqdm.github.io/)

2. **Running the Code:**  
   - Ensure the source file is in your project directory.
   - Open a terminal (or command prompt) in the project directory.
   - Ensure all dependancies are satisfied or use the provided requirements.txt file using:
     ```
     python3 -m pip install -r requirements.txt
     ```
   - Run the script using:
     ```
     python3 a_star_neeraj_kirti_dayanidhi.py
     ```
   - The program will:
     - Display the initial map with clearances (using matplotlib with proper x–y axes).
     - Prompt you for the step size.
     - Prompt you for the start and goal coordinates in the full-scale canvas coordinate system (i.e. x in [0,600) and y in [0,250)). Note that Θ (theta) should be provided in degrees (and should be a multiple of 30 degrees).
     - Verify that the start and goal are in free space; if either is inside an obstacle, you will be prompted to re-enter the coordinates.
     - Run the A\* search algorithm while displaying a progress bar.
     - Animate the exploration and optimal path generation using OpenCV.
     - Finally, display the final canvas (with the optimal path overlaid) using matplotlib.

## Design Decisions and Notes

- **Obstacle Space & Scaling:**  
  The map is constructed based on raw measurements (from Project 2) and then scaled to a 600×250 canvas. To ensure accurate obstacle representation, the coordinates are scaled carefully (using different scale factors for x and y) before applying clearance and robot radius inflation.

- **Action Set and Discretization:**  
  The robot’s motion is defined by five distinct actions. Discretization of the state is performed with fixed thresholds (0.5 units for x,y and 30° for theta). These thresholds help prevent duplicate nodes and manage the size of the search space.

- **Visualization:**  
  Visualization is split between dynamic (animated exploration and path drawing using OpenCV) and static displays (using matplotlib to show images with axes). OpenCV is used for real-time animations, while matplotlib is used to present images with coordinate axes for clarity.

- **Edge Cases:**  
  An edge case was noted when start/goal points are extremely close to the boundaries. In particular, when the start is near the top-right (e.g., (599,249) in full-scale), scaling issues might cause neighbor generation to fail. A small epsilon tolerance (or a documentation note) clarifies this limitation. This epsilon tolerance is currently commented out, but can be engaged by following the instructions provided in the code via the corresponding comments.

- **Performance Optimizations:**  
  Precomputed trigonometric values in the motion functions reduced redundant calculations, resulting in an approximate 10% speedup for typical scenarios (e.g., with a step size of 5).

- **Inputs Used For Testing**
  ```
  Step size = 5
  Start Point = 10,125,0
  Goal Point = 590,125,60
  ```

## Limitations

- **Boundary Edge Cases:**  
  Extremely high or low coordinates (very close to canvas boundaries) might occasionally cause the algorithm to fail to find a path due to discretization and rounding effects. This is documented here and in the source code comments.

- **Fixed Canvas Size:**  
  The assignment requires a canvas of 600×250. Resizing for display purposes is implemented separately using matplotlib and OpenCV’s resize, but the internal resolution remains fixed.

## Team Members

- **Neeraj Laul** – Directory ID: nslaul, UID: 120518973  
- **Kirti Kishore** – Directory ID: kiki1, UID: 120148286
- **Dayanidhi Kandade** – Directory ID: dotsv, UID: 120426711

## Files Included

- **a_star_neeraj_kirti_dayanidhi.py** – Source code for the A\* algorithm and visualization.  
- **README.md** – This file.  
- **ENPM661_Project03_Phase01.pdf** – Source code in a pdf format (for plagiarism check and submission guidelines).  
- **Animation Video (.mp4)** – A video recording of the node exploration and optimal path generation. 

## References

- [OpenCV Drawing Functions](https://pyimagesearch.com/2021/01/27/drawing-with-opencv/)  
- [Matplotlib Animation Guide](https://matplotlib.org/stable/gallery/animation/dynamic_image.html)  
- [Heapq Documentation](https://docs.python.org/3/library/heapq.html)
