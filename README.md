# DASF module for Seismic

[![Continuous Test](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/ci.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/ci.yaml)
[![Commit Check Policy](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/commit-check.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-seismic/actions/workflows/commit-check.yaml)

DASF Seismic is a module dedicated to seismic operations. The project contains 
datasets that are capable to manipulate seismic formats. It is also possible
to calculate the most common seismic atrtibutes and many other common
transformations.

This project is based on top of [dasf-core](https://github.com/lmcad-unicamp/dasf-core) 
and it also follows the scikit learn API. The project includes support to 
clustered operations by using Dask and/or GPU usage either.

### Containers

To install DASF Seismic using docker or singularity, you must in the go to the
`build/` directory and execute the command below directory according to your
build type: `cpu` or `gpu`. Notice that DASF uses [HPC Container Maker](https://github.com/NVIDIA/hpc-container-maker)
(HPCCM) to generate recipes for all sort of container types. You should install
HPCCM first, in order to generate them.

```bash
./build_container.sh --device <cpu|gpu>
```

You can also configure other parameters of the container if you want. Run `-h`
for further information. It includes the container backend: docker or
singularity.

## Install

The installation can be done using `poetry install` or `pip install` with 
wheel installed and it requires [**dasf-core**](https://github.com/discovery-unicamp/dasf-core)
installed first.

```bash
pip3 install .
```

# Attributes

For further revision of what attribute is implemented see the Documentation. 
The list of implemented attributes is following:

- [Texture Attributes](docs/attributes/texture.md)
