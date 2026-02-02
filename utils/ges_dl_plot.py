import matplotlib.pyplot as plt
import numpy as np
from ges_data_loader_a import GESDataLoader
import ges_data_loader_a

# Pfad zu deinem Ordner mit JSON + footage
#file_path = "/home/tore/Volume/640x480_norot_noh/"
file_path = "/home/tore/Volume/1000x1000_droidtest3/"
#file_path = "/home/tore/Volume/homog1/"

# DataLoader initialisieren
dl = GESDataLoader(file_path)

# ENU-Koordinaten
enu = dl.t

print("---- ENU Koordinaten ----")
for i, p in enumerate(enu):
    print(f"Frame {i}: {p}")

# Rotationsmatrizen
T_matrices = np.array(dl.T_matrices)
R_matrices = T_matrices[0:3,0:3]
print("\n---- Rotationsmatrizen ----")
for i, R in enumerate(R_matrices):
    print(f"Frame {i}:\n{R}\n")

# 4x4 T-Matrizen
print("\n---- 4x4 T-Matrizen ----")
for i, T in enumerate(T_matrices):
    print(f"Frame {i}:\n{T}\n")

# Flugbahn plotten (ENU)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(enu[:,0], enu[:,1], '-o', label="Flugbahn")
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title('Flugbahn in ENU-Koordinaten')
ax.grid(True)
ax.axis('equal')
ax.legend()
plt.show()
plt.savefig(file_path+"gt.png")

ges_data_loader_a.export_kitti_poses(T_matrices, file_path+"gt.txt")
