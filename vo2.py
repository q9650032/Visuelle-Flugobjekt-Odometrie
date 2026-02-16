import cv2
import numpy as np
import utils.ges_data_loader_a
from utils.ges_data_loader_a import GESDataLoader
from matplotlib import pyplot as plt


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

def create_mask(w, h):
    mask_w = int(w/4)
    mask_h = int(h/4)
    mask_img = np.ones((h,w),dtype=np.uint8)*255
    cv2.rectangle(mask_img,(w-mask_w,h-mask_h),(w,h),0,thickness=-1)
    return mask_img

def vo_KLT():
    MIN_FEATURES = 120
    #dl = GESDataLoader(r'/home/tore/Volume/TestSetup1/')
    dl = GESDataLoader(r'/home/tore/Volume/TestSetup2/')
    gt = np.array(dl.T_matrices)
    est = []
    w = dl.image_width
    h = dl.image_height
    T_world = np.eye(4)
    plt.figure(figsize=(6,6))

    mask_img = create_mask(w, h)
    #cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("Mask", mask_img)


    # Create some random colors
    color = tuple(np.random.randint(0,255,3).tolist())

    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)
    old_frame = cv2.imread(dl.image_files[0])

    p0 = cv2.goodFeaturesToTrack(
        img1,
        maxCorners=0,
        qualityLevel=0.1,
        minDistance=5,
        mask=mask_img
    )

    # Create a mask image for drawing purposes
    mask_draw = np.zeros_like(old_frame)

    cur_pose = np.eye(4)  # 2D Pose in X/Y
    open("klt_est2.txt", "w").close()

    for i in range(1, len(dl.image_files)-1):
        img2 = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(dl.image_files[i])

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            err = err[st==1]
            mask_err = err < 12.0
            good_new = good_new[mask_err]
            good_old = good_old[mask_err]

        if good_old.shape[0] >= 4:

            H, inliers = cv2.findHomography(
                good_old,
                good_new,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
        else:
            print(f"Frame {i}: zu wenige Punkte ({good_old.shape[0]})")
            continue

        if H is None:
            continue

        inliers = inliers.ravel().astype(bool)
        good_new = good_new[inliers]
        good_old = good_old[inliers]

        R_raw = H[0:2, 0:2]

        # Orthonormalisieren (sehr wichtig!)
        U, _, Vt = np.linalg.svd(R_raw)
        R = U @ Vt

        R = R.T

        # Translation
        t = H[0:2, 2]

        # Skalenkompensation
        scale = np.linalg.norm(R_raw[:,0])
        if scale > 1e-6:
            t /= scale

        t = -R @ t

        # SE(2)-Delta
        T_delta = np.eye(3)
        T_delta[0:2, 0:2] = R

        # KITTI
        T_local = np.eye(4)
        T_local[:2,:2] = R
        print(t.shape)
        T_local[:2, 3] = t

        cur_pose = cur_pose @ T_local
        T_world = T_world @ T_local
        est.append(T_world)

        kitti_line = np.zeros((13))
        kitti_line[0] = int(i)
        kitti_line[1:] = T_world[:3,:4].reshape(-1)
        print(kitti_line)

        with open("klt_est2.txt", "a") as f:
            np.savetxt(f, kitti_line.reshape(1, -1), fmt="%.6f")

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask_draw, (int(a), int(b)), (int(c), int(d)), color, 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame',img)

        img1 = img2.copy()
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    est = np.array(est)
    N = min(len(gt), len(est))
    gt = gt[:N]
    est = est[:N]

    gt_xyz = gt[:, :3, 3]
    est_xyz = est[:, :3, 3]

    s, R, t = umeyama_alignment(gt_xyz, est_xyz, with_scale=True)
    est_aligned = (s * (R @ est_xyz.T)).T +t
    plt.plot(gt[:,0,3], gt[:,1,3])
    plt.plot(est_aligned[:,0], est_aligned[:,1])
    plt.show()

if __name__ == "__main__":
    vo_KLT()

