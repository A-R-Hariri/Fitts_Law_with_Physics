# Physics-Based Fitts' Law Environment for Myoelectric Control

A 2D Fitts' Law cursor task with explicit physics for benchmarking continuous myoelectric control. Mass, inertia, damping, and velocity limits are modeled so controller comparisons reflect actuator dynamics, not idealized point-and-click. Built in PySide6 with logging, deterministic replay, and offline analysis.

## Why physics

Standard myocontrol Fitts environments use kinematic cursors that abstract away the inertia of real actuators. Controllers that look good kinematically often fail when driving a physical prosthetic arm. This environment closes that gap before hardware deployment, and virtual metrics were validated against the Refined Clothespin Relocation Test on a physical prosthesis.

## Components

- `Fitts.py`: PySide6 task environment with configurable mass, damping, velocity limits, target sets
- `collect.py`: real-time data collection during user trials
- `Replay.py`: deterministic replay of logged trials for offline controller comparison
- `models.py`: control policies and EMG decoders
- `main.py`: trial runner
- `Analysis.ipynb`: throughput, completion rate, overshoot, Fitts' throughput regression

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.
