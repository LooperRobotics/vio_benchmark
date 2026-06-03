FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# The ros:humble base image (4 years old) has an expired ROS2 GPG key.
# Ubuntu repos are still valid, so we install curl from there first
# (--allow-unauthenticated skips the broken ROS2 signature),
# then use curl to refresh the key properly.
RUN apt-get update --allow-insecure-repositories 2>/dev/null || true \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
        curl ca-certificates gnupg \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
       | gpg --dearmor -o /usr/share/keyrings/ros2-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros2-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       > /etc/apt/sources.list.d/ros2-latest.list \
    && rm -rf /var/lib/apt/lists/*

# cv_bridge is the only ROS package not in ros-base; everything else
# (rclpy, sensor_msgs, geometry_msgs, tf2_msgs) is already included.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# pupil-apriltags bundles libapriltag3 — no separate C library install needed.
RUN pip3 install --no-cache-dir \
        "numpy>=1.22,<2" \
        "scipy>=1.8" \
        "matplotlib>=3.5" \
        pyyaml \
        aprilgrid \
        evo

WORKDIR /vio_benchmark

# ros:humble already ships /ros_entrypoint.sh which sources ROS2 setup.bash
ENTRYPOINT ["/ros_entrypoint.sh"]
#CMD ["python3", "main.py"]
