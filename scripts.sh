function project::run {
  # This function requires you to build the tensorflow image first:
  # $ sudo docker build . -t tensorflow                                                                                                                                                             [INSERT]
  sudo docker run \
    --user $(id -u):$(id -g) \
    --runtime=nvidia \
    --gpus all \
    --interactive \
    --tty \
    --rm \
    -p 8888:8888 \
    --volume $PWD:/code \
    tensorflow $1
}
