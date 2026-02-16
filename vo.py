import cv2
import numpy as np
import utils.ges_data_loader_a
from utils.ges_data_loader_a import GESDataLoader
from scipy.optimize import least_squares
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

def vo_homography_rot():
    dl = GESDataLoader(r'/home/tore/Volume/TestSetup2/')
    #dl = GESDataLoader(r'/home/tore/Volume/TestSetup1/')
    gt = np.array(dl.T_matrices)
    w = dl.image_width
    h = dl.image_height
    K = dl.K
    T_world = np.eye(4)
    plt.figure(figsize=(6,6))
    mask_img = create_mask(w, h)

    det = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)

    # 2D Pose (SE2)
    cur_pose = np.eye(3)
    poses = []

    open("orb_est.txt", "w").close()

    for i in range(1, len(dl.image_files) - 1):
        img2 = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)

        # Feature Detection
        kp1, des1 = det.detectAndCompute(img1, mask_img)
        kp2, des2 = det.detectAndCompute(img2, mask_img)
        if des1 is None or des2 is None:
            img1 = img2
            continue

        # Matching + Lowe Ratio
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(matches) < 4:
            img1 = img2
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
        cv2.imshow('Matches',match_img)
        # Homographie
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            img1 = img2
            continue

        A = H[0:2,0:2]
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt

        R = R.T

        t_img = H[0:2, 2]
        scale = np.linalg.norm(A[:,0])
        if  scale > 1e-6:
            t = t_img / scale

        # Translation korrekt drehen
        t_world = -R @ t

        T_delta = np.eye(3)
        T_delta[0:2, 0:2] = R
        T_delta[0:2, 2] = t_world

        cur_pose = cur_pose @ T_delta

        #KITTI EXPORT
        T_local = np.eye(4)
        T_local[:2,:2] = R
        T_local[:2, 3] = t_world
        T_world = T_world @ T_local
        kitti_line = np.zeros((13))
        kitti_line[0] = int(i)
        kitti_line[1:] = T_world[:3,:4].reshape(-1)
        #print(kitti_line)
        with open("orb_est.txt", "a") as f:
            np.savetxt(f, kitti_line.reshape(1, -1), fmt="%.6f")

        poses.append(cur_pose)

        cv2.waitKey(1)

        img1 = img2
    cv2.destroyAllWindows()
    est = np.array(poses)
    N = min(len(gt), len(est))
    gt = gt[:N]
    est = est[:N]

    gt_xyz = gt[:, :3, 3]
    est_xyz = est[:, :3, 2]

    s, R, t = umeyama_alignment(gt_xyz, est_xyz, with_scale=True)
    est_aligned = (s * (R @ est_xyz.T)).T +t
    plt.plot(gt[:,0,3], gt[:,1,3])
    plt.plot(est_aligned[:,0], est_aligned[:,1])
    plt.show()



if __name__ == "__main__":
    vo_homography_rot()
