#!/usr/bin/env python3

from dasf.datasets import DatasetArray, DatasetLabeled
from dasf.datasets.download import DownloadGDrive

from dasf_seismic.datasets import DatasetSeismicType


class F3LabeledSkel(DatasetArray, DownloadGDrive):
    def __init__(self,
                 name,
                 filename,
                 google_file_id,
                 download=True,
                 root=None,
                 chunks="auto"):

        self.filename = filename

        self._subtype = DatasetSeismicType.migrated_volume

        DatasetArray.__init__(self, name=name, download=False, root=root,
                              chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)


class F3Labeled(DatasetLabeled):
    def __init__(self, download=False, root=None, chunks="auto"):
        name = "F3 Netherlands Labeled"
        self._subtype = DatasetSeismicType.migrated_volume

        self._train = F3Train(download=download, root=root, chunks=chunks)
        self._val = F3Labels(download=download, root=root, chunks=chunks)

        super().__init__(name=name, download=download, root=root, chunks=chunks)


class F3Train(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Train Seismic", "train_seismic.npy",
                         "1U_FW4EkIWT8zVnYY9I7nlQf4TCJSM6Cc",
                         download=download, root=root, chunks=chunks)


class F3Labels(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Train Labels", "train_labels.npy",
                         "1EJr491BUpRoD4yhLeiOIW-A15idsdLet",
                         download=download, root=root, chunks=chunks)


class F3Test1(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Test 1 Seismic", "test1_seismic.npy",
                         "15oOv9THHZXq-9z9tdvxq1fReMAj57-Md",
                         download=download, root=root, chunks=chunks)


class F3Test1Labels(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Test 1 Labels", "test1_labels.npy",
                         "1pb27vQnmQ_FlhcAd1MATQfhS0GxXsJTt",
                         download=download, root=root, chunks=chunks)


class F3Test2(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Test 2 Seismic", "test2_seismic.npy",
                         "1V4hpnnVwPY5dusoiu5g8qBq9_-YEr49f",
                         download=download, root=root, chunks=chunks)


class F3Test2Labels(F3LabeledSkel):
    def __init__(self, download=True, root=None, chunks="auto"):
        super().__init__("F3 Netherlands Test 2 Labels", "test2_labels.npy",
                         "1T9IJcAT058piPUxKZNqi0Db3yNRydTWF",
                         download=download, root=root, chunks=chunks)
