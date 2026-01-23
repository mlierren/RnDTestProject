"""
Unified Optimization-based Motion Capture Filtering

This module implements a gradient-based unified optimization approach
for motion capture data denoising.

Key features:
- All constraints (bone length, ROM, smoothness) in a single loss function
- PyTorch autograd for efficient gradient computation
- Adaptive confidence weighting for automatic noise detection
- Tremor suppression and direction consistency for smooth motion
"""
