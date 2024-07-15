Installation
============

.. _installation:

Software Dependencies
:::::::::::::::::::::

 - python==3.10.13
 - numpy==1.26.4
 - pandas==2.2.2
 - matplotlib==3.8.2
 - scanpy==1.10.1
 - squidpy==1.4.1
 - faiss==1.7.4
 - igraph==0.11.5
 - scikit-learn==1.5.0
 - scikit-misc==0.3.1
 - louvain==0.8.2
 - scipy==1.12.0
 - tqdm==4.66.1
 - pytorch==2.3.1+cu121
 - torch_geometric==2.5.0
 - rpy2==3.2.2
 - R==3.6.1
 - mclust==6.1.1


Setup by Docker (`Recommended`)
:::::::::::::::::::::::::::::::

1. Download the stSCI image from `DockerHub <https://hub.docker.com/repository/docker/hannshu/stsci>`_ and setup a container:

   .. code-block:: bash

      docker run --gpus all --name your_container_name -idt hannshu/stsci:latest

2. Access the container:

   .. code-block:: bash

      docker start your_container_name
      docker exec -it your_container_name /bin/bash

3. Write a python script to run stSCI

The anaconda environment for stSCI will be automatically activate in the container. The stSCI source code is located at ``/root/stSCI``, please run ``git pull`` to update the codes before you use.
All dependencies of stSCI have been properly installed in this container, including the mclust R package, and the conda environment stSCI will automatically activate when you run the container.

- Note: Please make sure ``NVIDIA Container Toolkit`` is properly installed on your host device. (Or follow this instruction to `setup NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ first)


Manually setup  
::::::::::::::

We suggest you to use the Docker to setup and run stSCI. If you want to manually setup stSCI, we recommend you to use `Anaconda <https://docs.anaconda.com/free/anaconda/install>`_ to build the runtime environment.

1. Clone this repository from Github:

   .. code-block:: bash

      git clone https://github.com/hannshu/stSCI.git

2. Download dataset repository:

   .. code-block:: bash

      git submodule init
      git submodule update

3. Build the Anaconda environment, the package version of the essential dependencies noted at Software Dependencies section.

4. Write a python script to run stSCI
