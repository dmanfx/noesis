# 16. Multi-View 3D Tracking (MV3DT) (DS8)

DeepStream 8 introduces Multi-View 3D Tracking (MV3DT) as a developer preview feature. This technology enables real-time 3D tracking of objects across multiple calibrated cameras, providing a more comprehensive understanding of a scene.

## Key Features

-   **Real-time 3D Tracking**: Tracks objects in 3D space using input from multiple cameras.
-   **Multi-Camera Fusion**: Fuses data from multiple camera views to create a cohesive 3D representation.
-   **BEVFusion and BEVHeight**: Utilizes a new model that combines BEVFusion and BEVHeight for accurate 3D bounding box detection.
-   **V2XFusion**: Supports Vehicle-to-Everything (V2X) fusion, allowing the pipeline to process data from cameras and LiDARs simultaneously.

## How it Works

The MV3DT pipeline processes data from multiple cameras and corresponding LiDARs. It uses a model pre-trained on the DAIR-V2X dataset to infer 3D bounding boxes. The results can be visualized with projected LiDAR point clouds and camera images.

## Getting Started

To get started with MV3DT, you need to:

1.  **Set up the Environment**: Use the `deepstream-triton` base container and install the necessary dependencies.
2.  **Download Models and Datasets**: Download the pre-trained V2XFusion model and the V2X-Seq-SPD example dataset.
3.  **Generate TensorRT Engine**: Run the provided scripts to generate the model's TensorRT engine file.
4.  **Run the Pipeline**: Use the `deepstream-3d-lidar-sensor-fusion` sample application with the provided configuration file.

For detailed instructions, refer to the [DeepStream-3D Multi-Modal V2XFusion Setup](https://docs.nvidia.com/metropolis/deepstream/8.0/text/DS_3D_MultiModal_Lidar_Camera_V2XFusion.html) documentation.
