IMAGE_NAME="surrogate-nes-image"

docker build -t $IMAGE_NAME . -f Dockerfile

if [ -z "$XAUTH" ]; then
    XAUTH=/tmp/.docker.xauth
fi

if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run \
    -it \
    --rm \
    --name surrogate-nes-container \
    --gpus all \
    --ipc host \
    --runtime=nvidia \
    -p 6012:6011 \
    -e DISPLAY=$DISPLAY \
    --shm-size=64g \
    -v "$(pwd)/../code":/app/code \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --volume="$XAUTH:$XAUTH" \
    -e XAUTHORITY=$XAUTH \
    $IMAGE_NAME /bin/bash
