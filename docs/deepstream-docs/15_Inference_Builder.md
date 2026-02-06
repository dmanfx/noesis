# 15. Inference Builder (DS8)

DeepStream 8 introduces the Inference Builder, an open-source tool designed to automate the creation of end-to-end inference pipelines from a YAML configuration file. This tool simplifies the deployment of AI models by generating standalone Python applications or microservices.

## Key Features

-   **YAML-based Pipeline Definition**: Define your entire inference pipeline, including models, inputs/outputs, and backend inference engines in a single YAML file.
-   **Multi-Framework Support**: Supports multiple AI frameworks.
-   **Custom Logic Integration**: Allows for the seamless integration of custom pre-processing and post-processing code.
-   **Deployable Output**: Can package the entire pipeline as a deployable container image, a standalone Python application, or a microservice with an OpenAPI specification.

## How it Works

The Inference Builder takes a YAML file as input and generates the necessary code to run the inference pipeline. This approach allows for rapid prototyping and deployment of complex pipelines without writing extensive boilerplate code.

## Getting Started

To get started with the Inference Builder, you need to:

1.  **Clone the Repository**: The tool is available on GitHub.
2.  **Install Dependencies**: It requires Python 3.12 and several other packages.
3.  **Configure Docker**: A proper Docker environment is needed to build and run the container images.
4.  **Use Sample Configurations**: The repository includes several sample configurations to help you get started.

For detailed instructions, refer to the [Inference Builder README](https://github.com/NVIDIA-AI-IOT/inference_builder/blob/main/README.md).
