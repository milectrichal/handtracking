import time
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Fingertip landmark indices
TIP_IDS = [4, 8, 12, 16, 20]

# Skeleton connections (white lines)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Across the knuckles
    (5, 9), (9, 13), (13, 17)
]

def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

def angle_deg(a, b, c):
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    dot = abx * cbx + aby * cby
    ab = math.hypot(abx, aby)
    cb = math.hypot(cbx, cby)
    if ab == 0 or cb == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / (ab * cb)))
    return math.degrees(math.acos(cosv))

def thumb_is_up(hand_lms) -> bool:
    wrist = hand_lms[0]
    index_mcp = hand_lms[5]
    middle_mcp = hand_lms[9]
    ring_mcp = hand_lms[13]
    pinky_mcp = hand_lms[17]

    # Palm center (average of stable palm points)
    palm_cx = (wrist.x + index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 5.0
    palm_cy = (wrist.y + index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 5.0

    class P:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    palm_center = P(palm_cx, palm_cy)

    thumb_mcp = hand_lms[2]
    thumb_ip  = hand_lms[3]
    thumb_tip = hand_lms[4]

    # Straightness of thumb (angle MCP–IP–TIP)
    straight = angle_deg(thumb_mcp, thumb_ip, thumb_tip)

    # Tip should be farther from palm than IP to avoid counting folded thumb
    tip_farther_than_ip = dist2(thumb_tip, palm_center) > dist2(thumb_ip, palm_center) + 0.0004

    return (straight > 160.0) and tip_farther_than_ip

def count_fingers(hand_lms) -> int:
    fingers_up = 0

    if thumb_is_up(hand_lms):
        fingers_up += 1

    # (tip_id, pip_id)
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_id, pip_id in pairs:
        if hand_lms[tip_id].y < hand_lms[pip_id].y:
            fingers_up += 1

    return fingers_up

def main():
    # Create hand landmarker (MediaPipe Tasks API)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (permission? wrong index?)")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                total_fingers += count_fingers(hand)

                # Draw skeleton (white)
                for a, b in HAND_CONNECTIONS:
                    ax, ay = to_px(hand[a], w, h)
                    bx, by = to_px(hand[b], w, h)
                    cv2.line(frame, (ax, ay), (bx, by), (255, 255, 255), 3)

                # Draw joints (red)
                for lm in hand:
                    x, y = to_px(lm, w, h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Highlight fingertips (purple with red center)
                for tip_id in TIP_IDS:
                    x, y = to_px(hand[tip_id], w, h)
                    cv2.circle(frame, (x, y), 16, (255, 0, 255), -1)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Show finger count
        cv2.putText(
            frame, f"Fingers: {total_fingers}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3
        )

        cv2.imshow("Hand Tracker + Finger Count (Tasks)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
