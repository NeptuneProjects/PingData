# PingData

`PingData` is a lightweight Python package for reading [HumminBird](https://humminbird.johnsonoutdoors.com) metadata/sonar files and writing them to user-friendly formats, like `csv`, `HDF5`, `npy`, or `mat`.
The package is a fork of Cameron Bodine's `PINGMapper` project, which provides far more comprehensive (and complex) workflows than this package.
This package is completely unopinionated, and provides only raw metadata and sonar data (in integer format) with no post-processing.

## Installation

You can install `PingData` using pip:

```bash
pip install pingdata
```

## Usage

Top-level functions are provided in the `cli.py` module.
Conversion can be accomplished directly from the command line.
Consider an input metadata file, `path/to/Rec00001.DAT`:
```bash
pingdata path/to/Rec00001.DAT
```
In this case, both metadata and sonar data will be written to a directory with the same stem as the input file, i.e., `path/to/Rec00001/*`.
To specify an output directory, use the `-o` flag:
```bash
pingdata path/to/Rec00001.DAT -o custom/path
```
Data will be saved to `custom/path/Rec00001/*`.

Water temperature and sound speed can be optionally passed:
```bash
pingdata <input_file> -t 14.6 -s 1510.3
```

The output format can be specified with the `-f` flag:
```bash
pingdata <input_file> -f mat
```
The available formats are:
- `hdf5`
- `mat`
- `npy`

Regardless of the format chosen, metadata will always be written to a `.csv` file located in the same directory as the other outputs.