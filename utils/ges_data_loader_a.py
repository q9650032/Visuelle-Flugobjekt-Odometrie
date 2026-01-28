# TODO TEST AND VERIFY

import numpy as np
import os
from pathlib import Path
import json
import math

class GESDataLoader():
    def __init__(self, file_path):

        self.file_path = Path(file_path)

        footage = self.file_path / "footage"
        self.image_files = sorted(
            str(p.resolve())
            for p in footage.iterdir()
            if p.is_file()
        )
        self.json_file_name = self.file_path.stem+'.json'
        self.json_file = self.file_path / Path(self.json_file_name)

        self.image_width = None
        self.image_height = None
        self.vertical_fov = None

        self.lla = np.empty((0,3))
        self.ecef = np.empty((0,3))
        self.t = np.empty((0,3))  # ENU
        self.R_matrices = None
        self.T_matrices = []

        self.json_to_pose()
        #self.json_to_enu()
        self.K = self.K_from_vertical_fov()
        self.P = np.hstack(
            (self.K, np.zeros((3, 1), dtype=np.float32)))

    def Rx(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    def Ry(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    def Rz(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def json_to_pose(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)

        self.image_width = data['width']
        self.image_height = data['height']

        origin_frame = data["cameraFrames"][0]
        self.vertical_fov = origin_frame['fovVertical']

        # Ursprung für lokale ENU-Koordinaten (Trackpoint)
        origin = np.array([
            origin_frame["position"]["x"],
            origin_frame["position"]["y"],
            origin_frame["position"]["z"]
        ])

        for frame in data["cameraFrames"]:
            # --- LLA speichern ---
            lat = frame["coordinate"]["latitude"]
            lon = frame["coordinate"]["longitude"]
            alt = frame["coordinate"]["altitude"]
            self.lla = np.vstack([self.lla, np.array([lat, lon, alt])])

            # --- Positionsvektor ---
            x = frame["position"]["x"]
            y = frame["position"]["y"]
            z = frame["position"]["z"]
            t = np.array([x, y, z])

            # Lokale Position relativ zum Trackpoint
            t_local = t - origin

            # --- Rotationswinkel ---
            rx = np.deg2rad(frame["rotation"]["x"])
            ry = np.deg2rad(frame["rotation"]["y"])
            rz = np.deg2rad(frame["rotation"]["z"])

            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)

            # --- ENU-Basis ---
            R_enu_ecef = np.array([
                [-np.sin(lon_rad),               np.cos(lon_rad),            0],  # East
                [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],  # North
                [ np.cos(lat_rad)*np.cos(lon_rad),  np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]   # Up
            ])

            # --- Kamera-Rotation ECEF → ENU ---
            R_cam_ecef = self.Rz(rz) @ self.Ry(ry) @ self.Rx(rx)  # Intrinsic Z-Y-X
            R_cam_enu = R_enu_ecef @ R_cam_ecef

            # --- Lokale ENU-Koordinaten ---
            p_enu = R_enu_ecef @ t_local
            self.t = np.vstack((self.t, p_enu))

            # --- KITTI Achsenflip ---
            # ENU → KITTI (X=forward, Y=left, Z=up)
            A_kitti = np.array([
                [0, 0, -1],  # X_forward
                [-1, 0, 0],  # Y_left
                [0, 1, 0]    # Z_up
            ])
            R_kitti = R_cam_enu @ A_kitti
            p_kitti = A_kitti @ p_enu  # Position auch flippen

            # --- Pose-Matrix für KITTI ---
            T_kitti = np.hstack((R_kitti, p_kitti.reshape(3,1)))
            self.T_matrices.append(T_kitti)

    # -----------------------------
    # LLA <-> ECEF <-> ENU
    # -----------------------------
    def lla_to_ecef(self, lat, lon, alt):
        a = 6378137.0
        f = 1/298.257223563
        b = a * (1 - f)
        e2 = 1 - (b**2 / a**2)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + alt) * math.sin(lat_rad)
        return np.array([x, y, z])

    def ecef_to_enu(self, ecef, origin_ecef, origin_lat, origin_lon):
        lat_rad = math.radians(origin_lat)
        lon_rad = math.radians(origin_lon)
        R = np.array([
            [-math.sin(lon_rad), math.cos(lon_rad), 0],
            [-math.sin(lat_rad)*math.cos(lon_rad), -math.sin(lat_rad)*math.sin(lon_rad), math.cos(lat_rad)],
            [math.cos(lat_rad)*math.cos(lon_rad), math.cos(lat_rad)*math.sin(lon_rad), math.sin(lat_rad)]
        ])
        delta = ecef - origin_ecef
        return R @ delta

    def json_to_enu(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)

        self.image_width = data['width']
        self.image_height = data['height']

        # erster Frame als Origin für ENU
        origin_frame = data["cameraFrames"][0]
        origin_lat = origin_frame["coordinate"]["latitude"]
        origin_lon = origin_frame["coordinate"]["longitude"]
        origin_alt = origin_frame["coordinate"]["altitude"]
        self.vertical_fov = origin_frame['fovVertical']

        origin_ecef = self.lla_to_ecef(origin_lat, origin_lon, origin_alt)

        # ENU, ECEF, LLA für alle Frames
        for frame in data["cameraFrames"]:
            lat = frame["coordinate"]["latitude"]
            lon = frame["coordinate"]["longitude"]
            alt = frame["coordinate"]["altitude"]

            self.lla = np.vstack([self.lla, np.array([lat, lon, alt])])
            ecef = self.lla_to_ecef(lat, lon, alt)
            self.ecef = np.vstack([self.ecef, ecef])
            enu = self.ecef_to_enu(ecef, origin_ecef, origin_lat, origin_lon)
            self.t = np.vstack([self.t, enu])

    # -----------------------------
    # Kamera Rotation & KITTI T-Matrix
    # -----------------------------
    def compute_RT_from_enu(self):
        curr_poses = self.t[:-1, :]
        next_poses = self.t[1:, :]
        xymov = next_poses-curr_poses
        xymov[:, 2] = 0.0
        norm_mat = np.linalg.norm(xymov, axis=1)
        rot_is_zero = norm_mat < 1e-6

        xcam = xymov / norm_mat[~rot_is_zero, None]
        xcam[rot_is_zero] = np.array((1.0, 0.0, 0.0))
        zcam = np.zeros(xymov.shape)
        zcam[:,:] = np.array((0.0, 0.0, -1.0))
        ycam = np.cross(zcam, xcam)
        ycam /= np.linalg.norm(ycam, axis=1)[:, None]
        xcam = np.cross(ycam, zcam)

        self.R_matrices = np.zeros((len(xymov),3,3))
        self.R_matrices[:, :, 0] = xcam
        self.R_matrices[:, :, 1] = ycam
        self.R_matrices[:, :, 2] = zcam

        self.T_matrices = np.tile(np.eye(4), (len(xymov),1,1))
        self.T_matrices[:, :3, :3] = self.R_matrices
        self.T_matrices[:, :3, -1] = curr_poses


    # -----------------------------
    # Kamera-Intrinsics
    # -----------------------------
    def K_from_vertical_fov(self):
        fov_y_rad = math.radians(self.vertical_fov)
        fy = self.image_height / (2.0 * math.tan(fov_y_rad / 2.0))
        fx = fy * (self.image_width/self.image_height)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        K = np.array([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        return K

@staticmethod
def export_kitti_poses(T_matrices, output_path):
    """
    Exportiert KITTI-kompatible Ground-Truth-Posen.

    Format:
    - eine Zeile pro Frame
    - 12 Werte (3x4 Matrix)
    - row-major
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for T in T_matrices:
            T_3x4 = T[:3, :]  # obere 3x4 Matrix
            row = T_3x4.reshape(-1)  # row-major
            line = " ".join(f"{v:.9f}" for v in row)
            f.write(line + "\n")

    print(f"[OK] KITTI-Pose-Datei geschrieben: {output_path}")
