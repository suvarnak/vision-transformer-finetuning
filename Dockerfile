# Use the official NVIDIA CUDA 12.1 image as a base
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

# Install Python 3.10 and pip3
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch and Transformers libraries
RUN pip3 install torch torchvision transformers jupyter

# Expose the port for Jupyter Notebook (optional)
EXPOSE 8888

# Set the default command to run when the container starts
CMD ["bash"]