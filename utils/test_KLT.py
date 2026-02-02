import numpy as np
import matplotlib.pyplot as plt


def trajectory_length(traj):
    """
    traj: (N,3) array of positions
    returns: total trajectory length in meters
    """
    diffs = traj[1:] - traj[:-1]
    return np.sum(np.linalg.norm(diffs, axis=1))

def segment_ate(gt, est, segment_length=100.0):
    """
    gt, est: (N,3) aligned trajectories
    segment_length: meters
    """
    errors = []
    acc_dist = 0.0
    start_idx = 0

    for i in range(1, len(gt)):
        acc_dist += np.linalg.norm(gt[i] - gt[i-1])

        if acc_dist >= segment_length:
            gt_seg = gt[start_idx:i+1]
            est_seg = est[start_idx:i+1]

            if len(gt_seg) < 2:
                continue

            # local translation removal
            gt_local = gt_seg - gt_seg[0]
            est_local = est_seg - est_seg[0]

            err = np.sqrt(np.mean(np.sum((gt_local - est_local)**2, axis=1)))
            errors.append(err)

            start_idx = i
            acc_dist = 0.0

    return np.array(errors)


def end_point_error(gt, est):
    return np.linalg.norm(gt[-1] - est[-1])

def umeyama_alignment(X, Y, with_scale=True):
    """
    Align Y to X
    X: (N,3) ground truth
    Y: (N,3) estimate
    """
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    cov = Yc.T @ Xc / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        var_Y = np.mean(np.sum(Yc**2, axis=1))
        s = np.trace(np.diag(D) @ S) / var_Y
    else:
        s = 1.0

    t = mu_X - s * R @ mu_Y

    return s, R, t


def calc_scale(gt_file="gt.txt", est_file="est.txt"):
    """
    calcuates scale from ground truth file and estimation.
    Rotation is ignored.
    params:
    gt: ground truth file (KITTI-Format)
    est: estimation file KITTI-Format
    """

    gt_data = np.loadtxt(gt_file)
    est_data = np.loadtxt(est_file)

    timestamps = est_data[:,0]

    gt=np.zeros((len(timestamps),3))
    for i, t in enumerate(timestamps):
        gt[i,:] = gt_data[int(t),[3,7,11]]

    N = min(len(gt), len(est_data))
    gt = gt[:N]
    est = est_data[:N]
    est = est[:,[4,8,12]]

    # Startpunkt entfernen
    gt -= gt[0]
    est -= est[0]

    # Skala (least squares)
    scale = np.sum(gt * est) / np.sum(est**2)

    ### gt algigned
    s, R, t = umeyama_alignment(gt, est, with_scale=True)

    est_aligned = (s * (R @ est.T)).T + t
    gt_aligned  = gt  # bleibt unverändert

    # Trajektorie normieren
    gt_n = gt_aligned / np.linalg.norm(gt_aligned[-1] - gt_aligned[0])
    est_n = est_aligned / np.linalg.norm(est_aligned[-1] - est_aligned[0])

    shape_error = np.mean(np.linalg.norm(gt_n - est_n, axis=1))
    print("SHAPE ERROR:", shape_error)

    errors = gt_aligned - est_aligned
    ate_rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    print("ATE RMSE:", ate_rmse)

    traj_len = trajectory_length(gt_aligned)

    plt.figure(figsize=(6,6))
    plt.plot(gt_aligned[:,1], -gt_aligned[:,0], label="GT")
    plt.plot(est_aligned[:,1], -est_aligned[:,0], label="EST aligned")
    plt.axis("equal")
    plt.legend()
    plt.title("TEST KLT")
    plt.savefig("test_KLT.png")

    # End Point Error
    epe = end_point_error(gt_aligned, est_aligned)
    print(f"EPE: {epe:.2f} m")

    segment_len = 100.0
    segment_ate_errors = segment_ate(gt_aligned, est_aligned, segment_len)
    print("segment ate errors ",np.mean(segment_ate_errors))

    ate_percent = 100/traj_len * ate_rmse
    segment_ate_percent = 100/traj_len * np.mean(segment_ate_errors)

    with open("test_klt_results.txt", "w") as f:
        f.write(f"segment ate errors (mean, segment length = {segment_len}): {np.mean(segment_ate_errors)}m, {segment_ate_percent}%\n")
        f.write(f"ate error: {ate_rmse}m, {ate_percent}%\n")
        f.write(f"shape error: {shape_error}\n")
        f.write(f"trajectory length: {traj_len}\n")

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(gt_aligned[:,0], gt_aligned[:,1], '--b', label="Ground Truth")
    ax.plot(est_aligned[:,0], est_aligned[:,1], '-r', label="Estimated")
    #ax.set_xlabel('[m]')
    #ax.set_ylabel('[m]')
    ax.set_title('Trjektorie GT vs. KLT Estimated')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.savefig("test_KLT.png")

    plt.show()
    plt.close()


calc_scale("/home/tore/Volume/1000x1000_droidtest3/gt.txt", "/home/tore/Volume/github/q9650032/Visuelle-Flugobjekt-Odometrie/klt_est.txt")

