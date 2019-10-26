function project::run {
  # This function requires you to build the tensorflow image first:
  # $ sudo docker build . -t tensorflow
  sudo docker run \
    --name project \
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


function project::tensorboard {
  sudo docker exec \
    --interactive \
    --tty \
    --user $(id -u):$(id -g) \
    --workdir /code \
    --detach \
    project \
    tensorboard \
      --port 6006 \
      --host 0.0.0.0 \
      --logdir /code/.cache/tensorboard \

  echo 'TensorBoard available at: http://0.0.0.0:6006/'
}
