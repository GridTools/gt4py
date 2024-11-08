#!/bin/bash
CUDA_DRIVER_VERSION=$1

echo "Installing CUDA driver version $CUDA_DRIVER_VERSION"
apt-get -yqq install --no-install-recommends kmod wget
wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${CUDA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${CUDA_DRIVER_VERSION}.run
chmod +x NVIDIA-Linux-x86_64-${CUDA_DRIVER_VERSION}.run
./NVIDIA-Linux-x86_64-${CUDA_DRIVER_VERSION}.run -s -q -a \
    --no-nvidia-modprobe \
    --no-abi-note \
    --no-kernel-module \
    --no-distro-scripts \
    --no-opengl-files \
    --no-wine-files \
    --no-kernel-module-source \
    --no-unified-memory \
    --no-drm \
    --no-libglx-indirect \
    --no-install-libglvnd \
    --no-systemd
rm ./NVIDIA-Linux-x86_64-${CUDA_DRIVER_VERSION}.run