import cv2
import numpy as np
import time

# ==========================================
# 1. User Configuration
# ==========================================
print("The program is detected by python")
#TARGET_ID = 0           # Targeted ArUco ID (the one we need to track)
#TARGET_Z = 0.5          # Target distance：50 cm (0.5 m)
#TARGET_X = 0.0          # Target horizon：0.0 (the center of the Aurco)
MARKER_SIZE = 0.05      # The Aruco code length size

# P-Controller gain coefficient (reaction sensitivity)
K_linear = 0.5
K_angular = 1.0

# Intrinsic matrix in the camera, transforming from 3D to 2D
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))

# ==========================================
# 2. Camera initialization and Aurco module
# ==========================================
# The type of Aruco is 4X4, 50
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Initialization for the camera
print("Starting the camera, waiting......")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1) # one second delay for a adaption

if not cap.isOpened():
    print("❌ Camera cannot be detected")
    exit()

print("✅ Camera light up successfully")

# ==========================================
# 3. Main loop
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # turn color to gray for faster
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # label detection
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # if the label is detected and the targeted id is contained
    if ids is not None:
        # pose estimation and position calculation
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        
        for i, marker_id in enumerate(ids.flatten()):
            # get the x and z coordinate of current label
            current_x = tvecs[i][0][0]
            current_z = tvecs[i][0][2]
            
            # --- Different actions for different labels ---
            
            if marker_id == 0:
                # 【ID 0】 
                target_z, target_x = 0.5, 0.0
                error_x, error_z = current_x - target_x, current_z - target_z
                
                if abs(error_x) > 0.02 or abs(error_z) > 0.05:
                    cmd_wz = np.clip(-K_angular * error_x, -0.5, 0.5)
                    cmd_vx = np.clip(K_linear * error_z, -0.2, 0.2)
                    print(f"🔴 [ID 0 angle adjusting] v_x={cmd_vx:+.2f}, w_z={cmd_wz:+.2f}")
                else:
                    print("🎯 [ID 0 Targeted] Ready to launch！")
            
            elif marker_id == 1:
                # 【ID 1】 
                target_z, target_x = 0.2, 0.0
                error_x, error_z = current_x - target_x, current_z - target_z
                
                if abs(error_x) > 0.01 or abs(error_z) > 0.02:
                    cmd_wz = np.clip(-K_angular * error_x, -0.3, 0.3) # 轉慢一點
                    cmd_vx = np.clip(K_linear * error_z, -0.1, 0.1) # 走慢一點
                    print(f"🔵 [ID 1 Connecting] v_x={cmd_vx:+.2f}, w_z={cmd_wz:+.2f}")
                else:
                    print("✅ [ID 1 Connection completed]")

            elif marker_id == 2:
                print("🟡 [ID 2 Detecting] v_x=0.00, w_z=-0.50")

            else:
                pass
            cv2.aruco.drawDetectedMarkers(frame, corners)

        # 3D coordinate (red=X, green=Y, blue=Z)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)

    # showing the camera image
    cv2.imshow("AMR Visual Servoing Test", frame)
    
    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# release the camera
cap.release()
cv2.destroyAllWindows()