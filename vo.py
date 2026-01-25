import cv2
import numpy as np
from utils.ges_data_loader_a import GESDataLoader

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


def vo_homography():
    dl = GESDataLoader(r'/home/tore/Volume/DSO-TEST/')
    gt = np.array(dl.T_matrices)
    w = dl.image_width
    h = dl.image_height

    mask_img = create_mask(w, h)
    #cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("Mask", mask_img)

    det = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Trajektorie vorbereiten
    x = (gt[:,1,3]/10 + 500)
    y = ((gt[:,2,3]*-1)/10 + 600)
    cv2.namedWindow("Trajectory")
    traj = np.zeros((800, 1000, 3), np.uint8)
    for i in range(len(gt)):
        cv2.circle(traj, (int(x[i]), int(y[i])), 1, (100,0,0), 2)

    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)
    cur_pose = np.eye(3)  # 2D Pose in X/Y
    last_keyframe_idx = 0

    for i in range(1, len(dl.image_files)-1):
        img2 = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)

        # Keyframe Auswahl
        delta_t = np.linalg.norm(gt[i][:3,3] - gt[last_keyframe_idx][:3,3])
        if i % 3 != 0 and delta_t < 0.1:
            img1 = img2
            continue
        last_keyframe_idx = i

        # Feature Detection
        kp1, des1 = det.detectAndCompute(img1, mask_img)
        kp2, des2 = det.detectAndCompute(img2, mask_img)
        if des1 is None or des2 is None:
            img1 = img2
            continue

        # Matcher + Lowe-Ratio
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        if len(good) < 4:
            img1 = img2
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Homography zwischen Keyframes
        H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            img1 = img2
            continue

        # Extrahiere Translation in X/Y
        tx = H[0,2]
        ty = H[1,2]

        # Akkumulation der 2D-Pose
        cur_pose = cur_pose @ np.array([[1,0,tx],[0,1,ty],[0,0,1]])

        # Trajektorie zeichnen
        t_curr = cur_pose[:2,2]
        x = t_curr[0]/10 + 500
        y = -t_curr[1]/10 + 600
        cv2.circle(traj, (int(x), int(y)), 1, (0,0,255), 2)
        cv2.imshow("Trajectory", traj)
        cv2.waitKey(1)

        img1 = img2

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vo_homography_rot():
    dl = GESDataLoader(r'/home/tore/Volume/DSO-TEST/')
    gt = np.array(dl.T_matrices)
    w = dl.image_width
    h = dl.image_height

    mask_img = create_mask(w, h)

    det = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # GT-Trajektorie
    x_gt = (gt[:,1,3]/10 + 500)
    y_gt = ((gt[:,2,3]*-1)/10 + 600)
    traj = np.zeros((800, 1000, 3), np.uint8)
    for i in range(len(gt)):
        cv2.circle(traj, (int(x_gt[i]), int(y_gt[i])), 1, (100,0,0), 2)

    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)

    # 2D Pose (SE2)
    cur_pose = np.eye(3)
    last_keyframe_idx = 0

    cv2.namedWindow("Trajectory")

    for i in range(1, len(dl.image_files) - 1):
        img2 = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)

        # Keyframe Auswahl
        delta_t = np.linalg.norm(gt[i][:3,3] - gt[last_keyframe_idx][:3,3])
        if i % 3 != 0 and delta_t < 0.1:
            img1 = img2
            continue
        last_keyframe_idx = i

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
        if len(good) < 4:
            img1 = img2
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Homographie
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            img1 = img2
            continue

        # Rotation (oberer linker 2x2 Block)
        R = H[0:2, 0:2]

        # Orthonormalisieren
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # Translation
        t = H[0:2, 2]

        # Skalenkorrektur
        scale = np.linalg.norm(R[:, 0])
        if scale > 1e-6:
            t = t / scale

        t = t/2.5

        # SE(2)-Delta
        T_delta = np.eye(3)
        T_delta[0:2, 0:2] = R
        T_delta[0:2, 2] = t

        # Pose akkumulieren
        cur_pose = cur_pose @ T_delta

        # ==============================
        # Trajektorie zeichnen
        # ==============================

        t_curr = cur_pose[0:2, 2]
        x = t_curr[0] / 10 + 500
        y = -t_curr[1] / 10 + 600

        cv2.circle(traj, (int(x), int(y)), 1, (0, 0, 255), 2)
        cv2.imshow("Trajectory", traj)
        cv2.waitKey(1)

        img1 = img2

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vo_homography_rot()
