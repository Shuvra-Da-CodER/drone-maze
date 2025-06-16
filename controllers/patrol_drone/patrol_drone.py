from controller import Robot, Camera, GPS
from ultralytics import YOLO
import numpy as np
import cv2
import math
import os

# === INIT ===
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
width = camera.getWidth()
height = camera.getHeight()
aspect_ratio = height / width
fov_hori = camera.getFov()  # Horizontal FOV in radians
fov_vert = fov_hori * aspect_ratio

gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Initialize motors (Crazyflie has 4 propellers)
motors = [robot.getDevice(f"propeller{i}") for i in range(1, 5)]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

model = YOLO("yolov8n.pt")  # Class 0 = person

# === MAZE GRID SETTINGS ===
origin_x = -15  # world x of top-left maze cell
origin_y = -15  # world y of top-left maze cell (altitude is z)
cell_size = 1.0  # size of each maze block (meters)

# === WAYPOINTS ===
waypoints = []
for row in range(15, -16, -3):  # y from +15 to -15, step -3
    if (row // 3) % 2 == 0:
        for col in range(-15, 16, 3):
            waypoints.append((col, row))  # (x, y)
    else:
        for col in range(15, -16, -3):
            waypoints.append((col, row))
waypoint_index = 0
waypoint_threshold = 0.5
Kp = 0.5

def pixel_to_angle(pixel_x, pixel_y):
    theta_x = ((pixel_x - width / 2) / width) * fov_hori
    theta_y = ((pixel_y - height / 2) / height) * fov_vert
    return theta_x, theta_y

def world_to_grid(x, y):
    col = int((x - origin_x) / cell_size)
    row = int((y - origin_y) / cell_size)
    return row, col

def move_to_waypoint(x_d, y_d, target_x, target_y):
    error_x = target_x - x_d
    error_y = target_y - y_d
    vx = Kp * error_x
    vy = Kp * error_y
    return vx, vy

def apply_motor_thrusts(vx, vy, target_z=5.0):
    current_pos = gps.getValues()
    error_z = target_z - current_pos[2]
    vz = 2.0 * error_z  # P-controller for altitude

    base_thrust = 68.5 + vz  # Base hover thrust
    roll_thrust = vx * 20
    pitch_thrust = -vy * 20

    motors[0].setVelocity(base_thrust - roll_thrust + pitch_thrust)
    motors[1].setVelocity(base_thrust + roll_thrust + pitch_thrust)
    motors[2].setVelocity(base_thrust + roll_thrust - pitch_thrust)
    motors[3].setVelocity(base_thrust - roll_thrust - pitch_thrust)

# === MAIN LOOP ===
while robot.step(TIME_STEP) != -1:
    img = camera.getImage()
    image_np = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 4))
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)

    results = model.predict(rgb_image, conf=0.5, classes=[0], verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            theta_x, theta_y = pixel_to_angle(center_x, center_y)
            x_d, y_d, z_d = gps.getValues()

            if math.tan(theta_y) == 0:
                continue

            d = z_d / math.tan(theta_y)  # horizontal ground distance using z as height
            dx = d * math.sin(theta_x)
            dy = d * math.cos(theta_x)

            person_x = x_d + dx
            person_y = y_d + dy

            grid_row, grid_col = world_to_grid(person_x, person_y)

            print(f"Detected person at grid cell: ({grid_row}, {grid_col})")

            with open("/tmp/person_coords.txt", "w") as f:
                f.write(f"{grid_row},{grid_col}")

            break

    # === Waypoint Navigation ===
    x_d, y_d, _ = gps.getValues()
    target_x, target_y = waypoints[waypoint_index]
    vx, vy = move_to_waypoint(x_d, y_d, target_x, target_y)

    apply_motor_thrusts(vx, vy, target_z=5.0)

    distance_to_wp = math.hypot(target_x - x_d, target_y - y_d)
    if distance_to_wp < waypoint_threshold:
        waypoint_index = (waypoint_index + 1) % len(waypoints)

    # Optional: Display for debugging
    # cv2.imshow("Surveillance View", rgb_image)
    # cv2.waitKey(1)
