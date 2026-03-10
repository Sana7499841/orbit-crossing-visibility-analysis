# Orbit Crossing and Visibility Analysis

This repository implements a **space-based sensor visibility analysis** for an orbiting object using **Two-Line Element (TLE) propagation with SGP4**.

The objective is to detect when a space object **crosses the field of view (FOV)** of a space-based tracker and when it becomes **visible/detectable** according to geometric and illumination constraints.

---

# Objective

Within the **24 hours following the tracker epoch**, the program determines:

1. **Crossing Events**  
   Times when the target object passes through the sensor field of view (geometry only).

2. **Visible / Detectable Events**  
   Times when the object satisfies all detection conditions:
   - Inside the tracker field of view
   - Sunlit (not in Earth's shadow)
   - Distance between tracker and object ≤ 1000 km

---

# System Model

### Target Object
- Propagated using **SGP4**
- Input provided as **Two-Line Element (TLE)**
- Output reference frame: **TEME**

### Space-Based Tracker
- Propagated using a **two-body Keplerian model**
- Orbital elements specified at epoch

### Sensor Model
- Full field of view: **30°**
- Half-angle: **15°**
- Sensor boresight direction aligned with **tracker velocity vector**

### Illumination Model
A **cylindrical Earth shadow model** is used:

The object is considered **sunlit** if it is not within Earth's shadow cylinder along the Sun–Earth direction.

### Detection Constraint
A visible event must satisfy:
crossing AND sunlit AND range ≤ 1000 km

# Repository Structure
orbit-crossing-visibility-analysis
│
├── q1_space_tracker_crossings.py
├── Space_Object_Crossing_and_Visibility_Analysis.pdf
├── README.md
└── .gitignore

The program will output:

- Crossing intervals
- Visible/detectable intervals

for the **24-hour analysis window**.

---

# Output

Example output format:
Crossing events (geometry only):
Start: 2025-09-01T03:12:40Z
End: 2025-09-01T03:13:20Z
Visible events:
Start: 2025-09-01T03:12:50Z
End: 2025-09-01T03:13:10Z

---

# Notes

- The SGP4 propagator outputs positions in the **TEME frame**.
- For this assignment, the tracker orbit is treated as being in an **inertial-like frame**, and frame transformation effects are neglected for short-duration analysis.

---

# Author

Upasana Panigrahi  
Aerospace Engineering
