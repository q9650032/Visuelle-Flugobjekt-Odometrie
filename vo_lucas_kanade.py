import cv2
import numpy as np
import utils.ges_data_loader_a
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


def grid_features(img, grid=4, maxCorners=200):
    h, w = img.shape
    pts = []
    for y in range(grid):
        for x in range(grid):
            roi = img[
                y*h//grid:(y+1)*h//grid,
                x*w//grid:(x+1)*w//grid
            ]
            corners = cv2.goodFeaturesToTrack(roi, maxCorners//(grid*grid), 0.01, 7)
            if corners is not None:
                corners[:,0,0] += x*w//grid
                corners[:,0,1] += y*h//grid
                pts.append(corners)
    return np.vstack(pts) if pts else None


def vo_KLT():
    MIN_FEATURES = 120
    dl = GESDataLoader(r'/home/tore/Volume/1000x1000_droidtest3/')
    gt = np.array(dl.T_matrices)
    w = dl.image_width
    h = dl.image_height
    T_world = np.eye(4)
    yaw_global = 0.0

    mask_img = create_mask(w, h)
    #cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("Mask", mask_img)


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (31, 31),
                      maxLevel = 4,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Create some random colors
    color = tuple(np.random.randint(0,255,3).tolist())

    # Trajektorie vorbereiten
    x = (-gt[:,1,3]/10 + 500)
    y = ((-gt[:,2,3])/10 + 600)
    cv2.namedWindow("Trajectory")
    traj = np.zeros((800, 1000, 3), np.uint8)


    old_gray = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)
    for i in range(len(gt)):
        cv2.circle(traj, (int(x[i]), int(y[i])), 1, (100,0,0), 2)

    old_gray = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)
    old_frame = cv2.imread(dl.image_files[0])
    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p0 = grid_features(old_gray)
    # Create a mask image for drawing purposes
    mask_draw = np.zeros_like(old_frame)

    cur_pose = np.eye(4)  # 2D Pose in X/Y
    open("klt_est.txt", "w").close()

    for i in range(1, len(dl.image_files)-1):
        frame_gray = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(dl.image_files[i])

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            err = err[st==1]
            mask_err = err < 12.0
            good_new = good_new[mask_err]
            good_old = good_old[mask_err]

        if good_old.shape[0] >= 4:
            M, inliers = cv2.estimateAffinePartial2D(
                good_old,
                good_new,
                method=cv2.RANSAC,
                ransacReprojThreshold=3)
            inliers = inliers.ravel().astype(bool)
            good_new = good_new[inliers]
            good_old = good_old[inliers]

        else:
            print(f"Frame {i}: zu wenige Punkte ({good_old.shape[0]})")
            continue

        theta_img = np.arctan2(M[1,0], M[0,0])
        yaw = -theta_img

        R_kitti = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0],
            [ np.sin(yaw),  np.cos(yaw), 0],
            [ 0,            0,           1]
        ])
        # Extrahiere Translation in X/Y
        tx_img = M[0,2]
        ty_img = M[1,2]
        if np.linalg.norm([tx_img, ty_img]) < 0.5:
            continue
        t_kitti = np.array([
            tx_img,    # links
            ty_img,     # vorwärts
            0
        ])

        T_local = np.eye(4)
        T_local[:3,:3] = R_kitti
        T_local[:3, 3] = t_kitti

        cur_pose = cur_pose @ T_local
        T_world = T_world @ T_local

        kitti_line = np.zeros((13))
        kitti_line[0] = int(i)
        kitti_line[1:] = T_world[:3,:4].reshape(-1)
        print(kitti_line)

        with open("klt_est.txt", "a") as f:
            np.savetxt(f, kitti_line.reshape(1, -1), fmt="%.6f")

        print(f"yaw [deg]: {np.degrees(yaw):.4f}")

        # Trajektorie zeichnen
        t_curr = cur_pose[:3,3]
        x = -t_curr[0]/25 + 500
        y = -t_curr[1]/25 + 600
        cv2.circle(traj, (int(x), int(y)), 1, (0,0,255), 2)
        cv2.imshow("Trajectory", traj)
        cv2.waitKey(1)

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask_draw, (int(a), int(b)), (int(c), int(d)), color, 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()

        if len(p0) < MIN_FEATURES:
            new_pts = grid_features(frame_gray)
            # new_pts = cv2.goodFeaturesToTrack(
            #     frame_gray,
            #     maxCorners=300,
            #     qualityLevel=0.01,
            #     minDistance=7,
            #     mask=mask_img
            # )
            if new_pts is not None:
                p0 = np.vstack((p0, new_pts))


    cv2.destroyAllWindows()

if __name__ == "__main__":
    vo_KLT()

