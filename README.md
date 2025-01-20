
# Quadrotor Formation using Model Predictive Control

  

## Overview

This project implements Model Predictive Control (MPC) for quadrotor formation control, with specific focus on maintaining stability and performance even after rotor failure. The implementation utilizes the Acados framework for efficient real-time optimization.

  

## Prerequisites

- Python 3.7 or higher
- CMake
- C/C++ compiler
- Git

## Installation
  
### Building Acados from Source

```bash
# Clone the repository and its submodules:
git  clone  https://github.com/acados/acados.git
cd  acados
export ACADOS_SOURCE_DIR=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
git  submodule  update  --recursive  --init

# Build and install Acados:
mkdir  -p  build
cd  build
cmake  -DACADOS_WITH_QPOASES=ON  ..
make  install  -j4
  
#  Install the Python interface:
cd  $ACADOS_SOURCE_DIR
pip  install  -e  interfaces/acados_template
pip install pyquaternion
```

## Project Structure


### Core Components
  
-  `main.py`: Driver code containing primary control functions
-  `quadrotor.py`: Quadrotor dynamics implementation
-  `controller.py`: MPC solver interface and control computation
-  `utils.py`: Utility functions supporting core functionality

  

### Key Functions

  

The project implements four primary control modes:
1.  `move2Goal(goal_coordinates)`
- Navigates the quadrotor to specified target coordinates
- Parameters: `goal_coordinates` (x, y, z position)  

3.  `stableHover(hover_coordinates)`
- Achieves stable hovering at specified position
- Parameters: `hover_coordinates` (x, y, z position)

4.  `basicNavigation()`
- Demonstrates trajectory tracking capabilities
- Handles complex paths with three-motor operation
- No parameters required

5.  `controlledLanding(initial_position)`
- Executes controlled descent and landing
- Parameters: `initial_state` (all state variables)
 
## Usage

To run the simulation, execute the main script and specify the function to execute. For move2Goal:

General Usage
```bash
python main.py <function_name> 
```
Example Commands

1.    To run the move2Goal function:

```bash
python main.py move2Goal 
```

2. To run other functions, such as StableHover, BasicNavigation, or ControlledLanding:

```bash
python main.py StableHover
python main.py BasicNavigation
python main.py ControlledLanding
```

## Features


- Real-time MPC implementation using Acados
- Robust control under rotor failure conditions
- Multiple control modes for different scenarios
- Trajectory tracking with degraded actuator capabilities

  

## Notes

  

- The `<acados_root>` path must be the absolute path from `/home/`
- Ensure all shared libraries (libacados.so, libblasfeo.so, libhpipm.so) are properly linked
- The system has been tested with three-motor operation for degraded performance scenarios

