# NVIDIA DeepStream SDK - Distilled Documentation

Welcome to the project-specific, distilled documentation for the NVIDIA DeepStream SDK. This collection of documents is designed to be the primary source of truth for any development work involving DeepStream within this project.

The information contained here has been carefully extracted and summarized from the official NVIDIA DeepStream documentation, with a specific focus on our use case: a dGPU-based deployment on Ubuntu. Irrelevant information (e.g., for Jetson or WSL platforms) has been omitted to provide a clear and focused learning path.

## Table of Contents

This documentation is organized into several key areas. It is recommended to start with the `01_Core_Concepts.md` to get a foundational understanding of DeepStream.

1.  [**Core Concepts & Architecture**](./01_Core_Concepts.md)
    *   Understanding the DeepStream pipeline.
    *   High-level architecture.
    *   Application development workflow.

2.  [**GStreamer Plugins**](./02_GStreamer_Plugins.md)
    *   A comprehensive guide to the GStreamer plugins relevant to our dGPU setup.
    *   Details on what each plugin does, its key properties, and its outputs.

3.  [**Sample Applications Guide**](./03_Sample_Applications.md)
    *   A curated list of C/C++ and Python sample applications.
    *   Summaries of what each sample demonstrates to quickly find relevant examples.

4.  [**Using Custom Models**](./04_Custom_Models.md)
    *   Instructions and best practices for integrating custom AI models into DeepStream pipelines.

5.  [**Performance Tuning & Optimization**](./05_Performance.md)
    *   Techniques and configurations for optimizing pipeline performance on dGPU.

6.  [**Troubleshooting & FAQ**](./06_Troubleshooting.md)
    *   A collection of common issues, solutions, and frequently asked questions for dGPU on Ubuntu.

7.  [**3D DeepStream**](./07_3D_DeepStream.md)
    *   Documentation on the 3D capabilities of DeepStream that may be relevant for future use cases. 