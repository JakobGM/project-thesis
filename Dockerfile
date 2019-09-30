FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3 AS tensorflow

LABEL maintainer="jakobgm@gmail.com"
LABEL description="Tensorflow v2 GPU-enabled image with jupyterlab"

# Copy requirements into container and install all requirements
COPY ./requirements.txt requirements.txt
COPY ./requirements-dev.txt requirements-dev.txt
COPY ./requirements-flake8.txt requirements-flake8.txt
RUN pip install -r requirements-dev.txt

# Custom configuration paths for jupyter lab
COPY ./.docker/jupyter/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
RUN mkdir -p /.local/share/jupyter
RUN chmod 777 /.local/share/jupyter
RUN mkdir -p /.jupyter
RUN chmod 777 /.jupyter

# Expose the port used by jupyter lab
EXPOSE 8888

# Copy over jupyter lab startup script and use it as the default container command
COPY ./.docker/jupyter/jupyter-entrypoint.sh /jupyter-entrypoint.sh

# Install vim keybindings for jupyter lab
# nodejs npm -> required by jupyterlab_vim extension
# pandoc texlive-xetex -> required for nbconvert PDF export
RUN apt-get update && apt-get install -y \
        nodejs npm \
        pandoc texlive-xetex
RUN jupyter labextension install jupyterlab_vim

# Add volume mount point for source code
RUN mkdir /code
VOLUME ["/code"]

# Fixes GPU OOM on newer NVidia RTX20XX cards
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"

# Start the jupyter lab server
CMD ["/jupyter-entrypoint.sh"]
