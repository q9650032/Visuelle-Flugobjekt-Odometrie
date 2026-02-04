import cv2
import numpy as np
import utils.ges_data_loader_a
from utils.ges_data_loader_a import GESDataLoader
from scipy.optimize import least_squares

def se2_from_pose(T):
    x = T[0,2]
    y = T[1,2]
    theta = np.arctan2(T[1,0], T[0,0])
    return np.array([x,y,theta])


def pose_from_se2(p):
    x,y,theta = p
    T = np.eye(3)
    c = np.cos(theta)
    s = np.sin(theta)
    T[:2,:2] = np.array([[c,-s],[s,c]])
    T[:2,2] = [x,y]
    return T
def ba_residuals(params, rel_poses):
    N = len(rel_poses) + 1
    poses = params.reshape((N,3))
    res = []

    for i in range(N-1):

        Ti = pose_from_se2(poses[i])
        Tj = pose_from_se2(poses[i+1])

        T_est = np.linalg.inv(Ti) @ Tj

        dx_est = T_est[0,2]
        dy_est = T_est[1,2]
        dtheta_est = np.arctan2(T_est[1,0], T_est[0,0])

        dx,dy,dtheta = rel_poses[i]

        res.extend([
            dx_est-dx,
            dy_est-dy,
            dtheta_est-dtheta
        ])

    return res

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
    dl = GESDataLoader(r'/home/tore/Volume/1000x1000_droidtest3/')
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

def run_sliding_BA(all_poses, rel_poses, window=15):

    if len(all_poses) < window+1:
        return all_poses

    rel_local = rel_poses[-window:]
    poses_local = all_poses[-(window+1):]

    init = np.array([se2_from_pose(p) for p in poses_local]).flatten()

    result = least_squares(
        ba_residuals,
        init,
        args=(rel_local,),
        max_nfev=20,
        loss='huber'
    )

    opt = result.x.reshape((-1,3))

    for k,p in enumerate(opt):
        all_poses[-(window+1)+k] = pose_from_se2(p)

    return all_poses

def vo_homography_rot():
    dl = GESDataLoader(r'/home/tore/Volume/1000x1000_droidtest3/')
    gt = np.array(dl.T_matrices)
    w = dl.image_width
    h = dl.image_height
    K = dl.K
    T_world = np.eye(4)
    mask_img = create_mask(w, h)
    rel_poses = []
    all_poses = [np.eye(3)]


    det = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # GT-Trajektorie
    x_gt = (gt[:,0,3]/10 + 500)
    y_gt = (-gt[:,1,3]/10 + 600)
    traj = np.zeros((800, 1000, 3), np.uint8)
    for i in range(len(gt)):
        cv2.circle(traj, (int(x_gt[i]), int(y_gt[i])), 1, (100,0,0), 2)

    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)

    # 2D Pose (SE2)
    cur_pose = np.eye(3)
    poses = []
    keyframes = []
    last_keyframe_idx = 0

    cv2.namedWindow("Trajectory")
    open("ges_homography_est.txt", "w").close()

    for i in range(1, len(dl.image_files) - 1):

        # windowed ba
        if len(all_poses) % 5 == 0:
            all_poses = run_sliding_BA(all_poses, rel_poses)

            # komplette BA Trajektorie zeichnen
            for T in all_poses:
                p = se2_from_pose(T)

                x = -p[0]/30 + 500
                y = -p[1]/30 + 600

                cv2.circle(traj,(int(x),int(y)),1,(0,255,0),2)

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

        theta = np.arctan2(H[1,0], H[0,0])
        yaw = -theta

        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        v = H[0:2,0]
        v = v / np.linalg.norm(v)

        R = np.array([
            [ v[0], -v[1]],
            [ v[1],  v[0]]
        ])

        # yaw_H = np.arctan2(H[1,0], H[0,0])
        # yaw_R = np.arctan2(R[1,0], R[0,0])
        # yaw_RT = np.arctan2(R.T[1,0], R.T[0,0])

        # print(yaw_H, yaw_R, yaw_RT)


        # if np.linalg.det(R) < 0:
        #     print("det R ", np.linalg.det(R))
        #     Vt[1,:] *= -1
        #     R = U @ Vt


        A = H[0:2,0:2]
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt

        R = R.T

        #yaw2 = np.arctan2(R[1,0], R[0,0])
        #print("H vs. R: ", yaw, yaw2)

        t_img = H[0:2, 2]

        # Translation korrekt drehen
        t_world = R @ t_img

        T_delta = np.eye(3)
        T_delta[0:2, 0:2] = R
        T_delta[0:2, 2] = t_world

        cur_pose = cur_pose @ T_delta

        # Relative Bewegung speichern
        dx = T_delta[0,2]
        dy = T_delta[1,2]
        dtheta = np.arctan2(T_delta[1,0], T_delta[0,0])

        rel_poses.append([dx,dy,dtheta])
        all_poses.append(cur_pose.copy())

        T_local = np.eye(4)
        T_local[:2,:2] = R
        T_local[:2, 3] = t_world
        T_world = T_world @ T_local
        kitti_line = np.zeros((13))
        kitti_line[0] = int(i)
        kitti_line[1:] = T_world[:3,:4].reshape(-1)
        #print(kitti_line)
        with open("ges_homography_est.txt", "a") as f:
            np.savetxt(f, kitti_line.reshape(1, -1), fmt="%.6f")

        # Trajektorie zeichnen

        opt_pose = all_poses[-1]
        p = se2_from_pose(opt_pose)

        x = -p[0]/30 + 500
        y = -p[1]/30 + 600
        cv2.circle(traj,(int(x),int(y)),1,(0,255,0),2)

        t_curr = cur_pose[0:2, 2]
        x = -t_curr[0] / 30 + 500
        y = -t_curr[1] / 30 + 600

        poses.append(cur_pose)
        keyframes.append(last_keyframe_idx)
        cv2.circle(traj, (int(x), int(y)), 1, (0, 0, 255), 2)
        cv2.imshow("Trajectory", traj)
        cv2.waitKey(1)

        img1 = img2

    #print(poses)
    #utils.ges_data_loader_a.export_kitti_poses(poses, "ges_est.txt", keyframes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ges_test():

    # Bilder laden

    #dl = GESDataLoader("/home/tore/Pictures/3/")
    dl = GESDataLoader("/home/tore/Volume/homog2/")
    K = dl.K
    T = np.array(dl.T_matrices)
    t = T[:,:,3]
    w = dl.image_width
    h = dl.image_height
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    trajectory = [(0.0, 0.0, 0.0)]  # X, Y, Yaw

    m_scale = 3000/dl.lla[0,2]

    mask = create_mask(w,h)

    orb = cv2.ORB_create(4000)

    cv2.namedWindow("Trajektorie")
    #traj = np.ones((800,800,3),np.uint8)*255
    traj = cv2.imread("/home/tore/Volume/github/bachelor/src/3_snapshot_01-03-2026_20_06_12.jpeg")
    x = t[:,2]/m_scale
    y = -t[:,1]/m_scale
    for i in range(len(t)):
        cv2.circle(traj, (int(y[i]+650),int(x[i]+590)), 1, (200,0,0), 2)

    x = 0
    y = 0
    img1 = cv2.imread(dl.image_files[0], cv2.IMREAD_GRAYSCALE)
    for i in range(len(dl.image_files)-1):
        img2 = cv2.imread(dl.image_files[i], cv2.IMREAD_GRAYSCALE)
        assert img1 is not None and img2 is not None


        # Feature Matching

        k1, d1 = orb.detectAndCompute(img1, mask)
        k2, d2 = orb.detectAndCompute(img2, mask)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(d1, d2, k=2)
        #matches = bf.match(d1, d2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 20:
            raise RuntimeError("Zu wenige Matches")

        good = sorted(good, key=lambda x: x.distance)

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])


        # Pixelverschiebung

        dx_pix = np.median(pts2[:,0] - pts1[:,0])
        dy_pix = np.median(pts2[:,1] - pts1[:,1])
        x+=dx_pix
        y+=dy_pix
        print("div pix x", dx_pix)
        print("div pix y", dy_pix)

        height = dl.lla[i,2]
        x_scale = height / fx
        y_scale = height / fy
        dx_m = (dx_pix * x_scale)/m_scale
        dy_m = (dy_pix * y_scale)/m_scale

        #dx_m = dx_pix * 0.4
        #dy_m = dy_pix * 0.3

        print("SCALE X",height/fx)
        print("SCALE Y",height/fy)


        # Yaw-Schätzung (ungefähr)
        angles = []
        for (p0, p1) in zip(pts1, pts2):
            v0 = p0 - np.array([cx, cy])
            v1 = p1 - np.array([cx, cy])
            ang = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
            angles.append(ang)
        yaw = np.median(angles)

        # Akkumulieren
        X_prev, Y_prev, Yaw_prev = trajectory[-1]
        X_new = X_prev + dx_m
        Y_new = Y_prev + dy_m
        Yaw_new = Yaw_prev + yaw
        trajectory.append((X_new, Y_new, Yaw_new))

        img1 = img2

        cv2.circle(traj, (int(x/10+650), int(y/10+590)), 1, (0, 0, 255), 2)
        cv2.imshow("Trajektorie", traj)
        cv2.waitKey(1)

if __name__ == "__main__":
    vo_homography_rot()
    #ges_test()
