from ase.build import bulk
from ase.io import write

# file to make Si.traj
# next dump_gpaw_hse_sr.py

atoms = bulk('Si', 'diamond', a=5.431)
write('./data/raw/Si.traj', atoms)
