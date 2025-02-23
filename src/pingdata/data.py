# -*- coding: utf-8 -*-

from pathlib import Path

class Metadata:
    def __init__(self):
        ...

    def read(self, path: Path):
        ...

    def write(self):
        ...


class SonarData:
    def __init__(self):
        ...

    def read(self):
        ...

    def write(self):
        ...