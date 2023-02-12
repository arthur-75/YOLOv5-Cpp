ARG ARCH=aarch64
ARG REPO=axisecp
ARG SDK=acap-native-sdk
ARG UBUNTU_VERSION=22.04
ARG VERSION=latest

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

# Set general arguments
ARG ARCH
ARG SDK_LIB_PATH_BASE=/opt/axis/acapsdk/sysroots/${ARCH}/usr
ARG BUILD_DIR=/opt/build

#-------------------------------------------------------------------------------
# Prepare build environment
#-------------------------------------------------------------------------------

# Install build dependencies for cross compiling OpenCV
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y -f --no-install-recommends \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------
# Build OpenCV libraries
#-------------------------------------------------------------------------------

ARG OPENCV_VERSION=4.6.0
ARG OPENCV_DIR=${BUILD_DIR}/opencv
ARG OPENCV_SRC_DIR=${OPENCV_DIR}/opencv-${OPENCV_VERSION}
ARG OPENCV_BUILD_DIR=${OPENCV_DIR}/build

WORKDIR ${OPENCV_DIR}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz | tar -xz
RUN curl -fsSL https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.tar.gz | tar -xz

RUN mv "opencv_contrib-4.6.0/modules/aruco"  "opencv-4.6.0/modules/"

RUN rm -rf "opencv_contrib-4.6.0"

WORKDIR ${OPENCV_BUILD_DIR}
ENV COMMON_CMAKE_FLAGS="-S $OPENCV_SRC_DIR \
        -B $OPENCV_BUILD_DIR \
        -D CMAKE_INSTALL_PREFIX=$SDK_LIB_PATH_BASE \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D WITH_OPENEXR=OFF \
        -D WITH_GTK=OFF \
        -D WITH_V4L=OFF \
        -D WITH_FFMPEG=OFF \
        -D WITH_GSTREAMER=OFF \
        -D WITH_GSTREAMER_0_10=OFF \
        -D BUILD_LIST=core,imgproc,video,aruco,dnn,aruco \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_OPENCV_APPS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_JPEG=ON \
        -D BUILD_PNG=OFF \
        -D WITH_JASPER=OFF \
        -D BUILD_PROTOBUF=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON "

# hadolint ignore=SC2086
RUN if [ "$ARCH" = aarch64 ]; then \
        # Source SDK environment to get cross compilation tools
        . /opt/axis/acapsdk/environment-setup* && \
        # Configure build with CMake
        # No need to set NEON and VFP for aarch64 since they are implicitly
        # present in an any standard armv8-a implementation.
        cmake \
        -D CMAKE_CXX_COMPILER=${CXX%-g++*}-g++ \
        -D CMAKE_CXX_FLAGS="${CXX#*-g++}" \
        -D CMAKE_C_COMPILER=${CC%-gcc*}-gcc \
        -D CMAKE_C_FLAGS="${CC#*-gcc}" \
        -D CMAKE_TOOLCHAIN_FILE=${OPENCV_SRC_DIR}/platforms/linux/aarch64-gnu.toolchain.cmake \
        $COMMON_CMAKE_FLAGS && \
        # Build and install OpenCV
        make -j "$(nproc)" install ; \
    else \
        printf "Error: '%s' is not a valid value for the ARCH variable\n", "$ARCH"; \
        exit 1; \
    fi

#-------------------------------------------------------------------------------
# Copy the built library files to application directory
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .
RUN mkdir lib && \
    cp -P ${OPENCV_BUILD_DIR}/lib/lib*.so* ./lib/

RUN mkdir -p util  && \
    curl -L -o util/labels.txt https://github.com/google-coral/test_data/raw/master/coco_labels.txt

COPY ./app/yolov5s.onnx ./util/
COPY ./app/classes.txt ./util/

#-------------------------------------------------------------------------------
# Finally build the ACAP application
#-------------------------------------------------------------------------------

RUN . /opt/axis/acapsdk/environment-setup* && acap-build . \
     -a 'util/classes.txt' -a 'util/yolov5s.onnx'
