# 17. MaskTracker (DS8)

DeepStream 8 introduces the MaskTracker, a new feature for real-time, pixel-level mask generation. This capability is essential for applications that require semantic segmentation and object tracking with high precision.

## Key Features

-   **Pixel-Level Mask Generation**: Generates detailed segmentation masks for objects in real-time.
-   **Enhanced Tracking Accuracy**: By using segmentation masks, the tracker can achieve more accurate results compared to traditional bounding box-based methods.

## How it Works

The MaskTracker likely integrates with the existing `nvtracker` plugin or works as a standalone component to process the output of a segmentation model. It would then associate the generated masks with tracked objects, providing persistent tracking of segmented objects across frames.

## Getting Started

As of the current DeepStream 8.0 release, the official documentation and code samples for the MaskTracker are not yet available due to some broken links in the documentation portal. Once the documentation is updated, it will provide detailed instructions on how to:

1.  **Configure a Segmentation Model**: Use a model that outputs pixel-level masks.
2.  **Enable the MaskTracker**: Configure the tracker to use the segmentation masks for tracking.
3.  **Access Mask Data**: Use probes to access the mask data for custom processing and visualization.

This document will be updated as soon as more information becomes available.
