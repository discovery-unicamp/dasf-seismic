#!/usr/bin/env python3

import os
from enum import Enum

from mdio import MDIOReader

try:
    import cupy as cp
except ImportError:
    pass

from dasf.datasets import Dataset
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import human_readable_size


class DatasetSeismicType(Enum):
    none = "none"
    cmp_gathers = "CMP Gathers"
    surface_seismic = "Surface Seismic"
    borehole_seismic = "Borehole Seismic"
    fourd_far_stack = "4D Far Stack"
    fourd_near_stack = "4D Near Stack"
    fourd_mid_stack = "4D Mid Stack"
    fourd_full_stack = "4D Full Stack"
    far_stack = "Far Stack"
    near_stack = "Near Stack"
    mid_stack = "Mid Stack"
    full_stack = "Full Stack"
    prestack_seismic = "Prestack Seismic"
    poststack_seismic = "Poststack Seismic"
    migrated_volume = "Migrated Volume"

    def __str__(self):
        return self.value


class DatasetMDIO(Dataset):
    def __init__(self,
                 name,
                 subtype=DatasetSeismicType.none,
                 download=True,
                 root=None,
                 chunks=None,
                 return_metadata=False):

        Dataset.__init__(self, name, download, root)

        self._subtype = subtype
        self._root_file = root

        self.chunks = chunks

        self.__return_metadata = return_metadata

        access_pattern = "012"  # default access pattern
        if chunks:
            if not isinstance(chunks, dict):
                raise Exception("Chunks should be a dict with format "
                                "{'iline': x, 'xline': y, 'twt': z}.")
            else:
                chunks_fields = ["iline", "xline", "twt"]
                access_pattern = ""
                for i, field in enumerate(chunks_fields):
                    if chunks.get(field) != -1:
                        access_pattern += str(i)
        self._access_pattern = access_pattern

        if root is not None:
            if not os.path.isdir(root):
                raise Exception("MDIO requires a root=filename.")

            self._root = os.path.dirname(root)

            self._metadata = self._load_meta()

    def _load_meta(self, backend="zarr"):
        assert self._root_file is not None, ("There is no temporary file to "
                                             "inspect")
        assert os.path.isdir(self._root_file), ("The root variable should "
                                                "be a MDIO file")

        self.__parse_chunks()

        return self.inspect_mdio_seismic_cube(backend=backend)

    def __read_mdio(self, backend="zarr"):
        return MDIOReader(self._root_file, backend=backend,
                          new_chunks=self.__chunks,
                          access_pattern=self._access_pattern,
                          return_metadata=self.__return_metadata)

    def _lazy_load_gpu(self):
        backend = "dask"
        self._metadata = self._load_meta(backend=backend)
        self._mdio = self.__read_mdio(backend=backend)
        self._data = self._mdio._traces.map_blocks(cp.asarray)
        return self

    def _lazy_load_cpu(self):
        backend = "dask"
        self._metadata = self._load_meta(backend=backend)
        self._mdio = self.__read_mdio(backend=backend)
        self._data = self._mdio._traces
        return self

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._mdio = self.__read_mdio()
        self._data = cp.asarray(self._mdio._traces)
        return self

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._mdio = self.__read_mdio()
        self._data = self._mdio._traces
        return self

    @task_handler
    def load(self):
        ...

    def inspect_mdio_seismic_cube(self, backend="zarr"):
        mdio = self.__read_mdio(backend=backend)

        mdio_size = 0
        if backend == "zarr":
            for z, v in mdio._traces.info_items():
                if z == "No. bytes":
                    mdio_size = int(v.split(' ')[0])
        elif backend == "dask":
            mdio_size = mdio._traces.nbytes
        else:
            raise ValueError(f"No valid {backend}.")

        return {
            'size': human_readable_size(mdio_size),
            'file': self._root_file,
            'subtype': str(self._subtype),
            'shape': mdio.shape,
            'samples': mdio.binary_header["Samples"],
            'interval': mdio.binary_header["Interval"],
            'block': {
               "chunks": self.__chunks
            }
        }

    def copy(self, url, **kwargs):
        if not hasattr(self, "_mdio"):
            raise Exception("Dataset must be loaded to be copied")
        self._mdio.copy(url, **kwargs)

    def __getitem__(self, idx):
        return self._data[idx]

    def get_iline(self, idx):
        if self._data is not None:
            return self._data.sel(iline=(idx)) \
                                     .transpose("twt", "xline").data
        return None

    def get_xline(self, idx):
        if self._data is not None:
            return self._data.sel(xline=(idx)) \
                                     .transpose("twt", "iline").data
        return None

    def get_slice(self, idx):
        if self._data is not None:
            return self._data.sel(twt=(idx), method="nearest") \
                                     .transpose("iline", "xline").data

    def __parse_chunks(self):
        assert self.chunks is None or isinstance(self.chunks, dict) or \
               self.chunks == "auto"

        if self.chunks is None or self.chunks == "auto":
            self.__chunks = self.chunks
            return

        dims = (64, 64, 64)  # MDIO default chunking

        chunks = []
        if "iline" in self.chunks:
            chunks.append(self.chunks["iline"])
        else:
            chunks.append(dims[0])

        if "xline" in self.chunks:
            chunks.append(self.chunks["xline"])
        else:
            chunks.append(dims[1])

        if "twt" in self.chunks:
            chunks.append(self.chunks["twt"])
        else:
            chunks.append(dims[2])

        self.__chunks = tuple(chunks)
