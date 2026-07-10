#!/usr/bin/env python3
"""Inspect LIBERO HDF5 structure."""
import h5py
f = h5py.File("/home/ubuntu/libero_data/libero_spatial/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5", "r")
d = f["data/demo_0/obs"]
print("obs keys:", list(d.keys()))
for k in d.keys():
    i = d[k]
    if hasattr(i, "shape"):
        print(f"  {k}: {i.shape} {i.dtype}")
    elif hasattr(i, "keys"):
        for sk in list(i.keys())[:3]:
            si = i[sk]
            if hasattr(si, "shape"):
                print(f"  {k}/{sk}: {si.shape} {si.dtype}")
for a in f.attrs: print(f"attr: {a}={f.attrs[a]}")
for a in f["data/demo_0"].attrs: print(f"demo attr: {a}={f['data/demo_0'].attrs[a]}")
f.close()
