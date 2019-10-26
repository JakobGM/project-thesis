function project::run {
  # This function requires you to build the tensorflow image first:
  # $ sudo docker build . -t tensorflow
  sudo docker run \
    --user $(id -u):$(id -g) \
    --runtime=nvidia \
    --gpus all \
    --interactive \
    --tty \
    --rm \
    -p 8888:8888 \
    -p 6006:6006 \
    --volume $PWD:/code \
    tensorflow $1
}
