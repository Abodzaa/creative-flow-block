import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random

# ===============================
# إعدادات عامة / General Settings
# ===============================
# عتبة السرعة (بكسل/ثانية) / Speed threshold (pixels per second)
SPEED_THRESHOLD = 45.0
WINDOW_TITLE = "Stars Trail + Portrait Mode - Fast = Stars & Normal View | Slow = Portrait (Dark Background)"

# =======================================
# تهيئة الصوت / Initialize sound with pygame
# =======================================
pygame.mixer.init()
speed_sound = None
sound_channel = None

# نحاول تحميل ملف WAV أولاً ثم MP3 إذا لم يوجد / Try WAV first, fallback to MP3
try:
    speed_sound = pygame.mixer.Sound("speed_whoosh.wav")
except Exception:
    try:
        speed_sound = pygame.mixer.Sound("speed_sound.mp3")
    except Exception:
        speed_sound = None

# ======================================
# تهيئة MediaPipe Pose / Initialize Pose
# ======================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    # فعّل قناع التجزئة للتأثير الضبابي / Enable segmentation for portrait mode
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==================================
# فتح الكاميرا / Open default webcam
# ==================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open default webcam (cv2.VideoCapture(0)).")

# تحسينات أداء بسيطة / Small performance tuning
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ========================================
# حالة التتبع والجسيمات / Tracking + particles
# ========================================
prev_center = None
prev_time = None
high_speed_active = False
particles = []
portrait_fade = 0.0  # تدرج سلس للتبديل بين الأوضاع / Smooth fade for mode transition

fps_prev_time = time.time()
fps_value = 0.0

# ==============================================================
# قائمة نقاط الجسم الرئيسية لتوليد النجوم / Key body landmarks
# ==============================================================
BODY_LANDMARKS_FOR_STARS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EAR.value,
]


def star_points(x, y, outer_r, inner_r, angle_deg=0.0):
    """
    إنشاء نقاط النجمة الخماسية الكلاسيكية (10 نقاط)
    Build classic 5-point star polygon points (10 points)
    """
    pts = []
    start_angle = np.deg2rad(angle_deg - 90.0)
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        theta = start_angle + i * (np.pi / 5.0)
        px = int(x + r * np.cos(theta))
        py = int(y + r * np.sin(theta))
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def draw_star(frame, x, y, size, color, angle=0.0, alpha=1.0):
    """
    رسم نجمة خماسية ممتلئة مع توهج أبيض خفيف.
    Draw a filled 5-point star with a subtle white glow.
    """
    if alpha <= 0:
        return

    h, w = frame.shape[:2]
    if x < -30 or x > w + 30 or y < -30 or y > h + 30:
        return

    # حساب نصف القطر الداخلي لنسبة شكل النجمة الكلاسيكي
    # Inner radius ratio for a classic star look
    outer_r = max(2, int(size))
    inner_r = max(1, int(outer_r * 0.45))

    pts_main = star_points(x, y, outer_r, inner_r, angle)
    pts_glow = star_points(x, y, int(outer_r * 1.35),
                           int(inner_r * 1.35), angle)

    # طبقة توهج منفصلة / Separate glow overlay
    glow_overlay = frame.copy()
    cv2.fillPoly(glow_overlay, [pts_glow],
                 (255, 255, 255), lineType=cv2.LINE_AA)

    # شدة التوهج تتغير مع alpha / Glow intensity scales with alpha
    glow_strength = 0.18 * float(alpha)
    cv2.addWeighted(glow_overlay, glow_strength, frame,
                    1.0 - glow_strength, 0, frame)

    # طبقة النجمة الأساسية / Main star overlay
    star_overlay = frame.copy()
    cv2.fillPoly(star_overlay, [pts_main], color, lineType=cv2.LINE_AA)

    main_strength = max(0.05, min(1.0, float(alpha)))
    cv2.addWeighted(star_overlay, main_strength, frame,
                    1.0 - main_strength, 0, frame)


def spawn_star_particles_from_body(lms, frame_w, frame_h, move_dx, move_dy, speed):
    """
    توليد جسيمات نجوم من نقاط متعددة على الجسم (الكتفين، الكوعين، المعصمين، الوركين، الركبتين، الكاحلين).
    Spawn star particles from multiple body landmarks (shoulders, elbows, wrists, hips, knees, ankles).
    النجوم تظهر من خلف اتجاه الحركة / Stars appear to come from behind movement direction.
    """
    global particles

    mag = float(np.hypot(move_dx, move_dy))
    if mag < 1e-6:
        return

    # اتجاه الحركة / Movement direction
    dir_x = move_dx / mag
    dir_y = move_dy / mag

    # سرعة جسيمات النجوم / Star particle speed
    trail_speed = float(np.clip(speed / 120.0, 1.2, 6.0))

    # عدد النجوم المولدة / Number of stars to generate
    total_stars = random.randint(10, 15)
    stars_per_landmark = max(1, total_stars // len(BODY_LANDMARKS_FOR_STARS))

    # إنشاء نجوم من كل نقطة في الجسم / Generate stars from each body landmark
    for lm_idx in BODY_LANDMARKS_FOR_STARS:
        if lm_idx >= len(lms):
            continue

        lm = lms[lm_idx]
        if lm.visibility < 0.4:
            continue

        # تحويل الإحداثيات المعيارية إلى بكسلات / Convert normalized to pixel coords
        px_orig = int(lm.x * frame_w)
        py_orig = int(lm.y * frame_h)

        # توليد عدة نجوم من هذه النقطة / Generate multiple stars from this landmark
        for _ in range(stars_per_landmark):
            # النجوم تأتي من خلف الشخص (عكس اتجاه الحركة) + انحراف عشوائي
            # Stars come from behind (opposite movement direction) + random spread
            offset_dist = random.uniform(15, 40)
            offset_x = -dir_x * offset_dist + random.uniform(-10, 10)
            offset_y = -dir_y * offset_dist + random.uniform(-10, 10)

            px = px_orig + offset_x
            py = py_orig + offset_y

            # السرعة: للخلف + انحراف صغير / Velocity: backward + small deviation
            vx = -dir_x * trail_speed + random.uniform(-0.7, 0.7)
            vy = -dir_y * trail_speed + random.uniform(-0.7, 0.7)

            life = random.randint(20, 40)
            size = random.randint(8, 18)

            # لون أصفر-أبيض لامع (BGR) / Bright yellow-white color (BGR)
            b = random.randint(170, 245)
            g = random.randint(225, 255)
            r = random.randint(235, 255)

            particles.append({
                "x": px,
                "y": py,
                "vx": vx,
                "vy": vy,
                "life": life,
                "max_life": life,
                "size": size,
                "color": (b, g, r),
                "angle": random.uniform(0, 360),
                "spin": random.uniform(-8.0, 8.0),
                "twinkle": random.uniform(0.0, np.pi * 2.0),
            })


def spawn_star_particles(center_x, center_y, move_dx, move_dy, speed):
    """
    توليد جسيمات نجوم خلف اتجاه الحركة (من مركز الجسم فقط - للتوافقية).
    Spawn star particles trailing behind movement direction (from body center - for compatibility).
    """
    global particles

    mag = float(np.hypot(move_dx, move_dy))
    if mag < 1e-6:
        return

    dir_x = move_dx / mag
    dir_y = move_dy / mag

    trail_speed = float(np.clip(speed / 120.0, 1.2, 6.0))

    count = random.randint(10, 15)
    for _ in range(count):
        vx = -dir_x * trail_speed + random.uniform(-0.9, 0.9)
        vy = -dir_y * trail_speed + random.uniform(-0.9, 0.9)

        px = center_x + random.uniform(-14, 14)
        py = center_y + random.uniform(-14, 14)

        life = random.randint(20, 40)
        size = random.randint(8, 18)

        b = random.randint(170, 245)
        g = random.randint(225, 255)
        r = random.randint(235, 255)

        particles.append({
            "x": px,
            "y": py,
            "vx": vx,
            "vy": vy,
            "life": life,
            "max_life": life,
            "size": size,
            "color": (b, g, r),
            "angle": random.uniform(0, 360),
            "spin": random.uniform(-8.0, 8.0),
            "twinkle": random.uniform(0.0, np.pi * 2.0),
        })


def update_and_draw_particles(frame):
    """
    تحديث الجسيمات ورسمها مع تلاشي ولمعان.
    Update particles and render with fading + twinkle.
    """
    global particles

    now = time.time()
    alive = []

    for p in particles:
        # تحديث الحركة / Motion update
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["angle"] += p["spin"]
        p["life"] -= 1

        if p["life"] <= 0:
            continue

        alpha = p["life"] / p["max_life"]

        # تأثير تلألؤ / Twinkling effect
        tw = 0.75 + 0.25 * np.sin(now * 12.0 + p["twinkle"])
        draw_size = max(2, int(p["size"] * tw))

        draw_star(
            frame,
            int(p["x"]),
            int(p["y"]),
            draw_size,
            p["color"],
            angle=p["angle"],
            alpha=alpha,
        )

        alive.append(p)

    particles = alive


def apply_portrait_effect(frame, segmentation_mask, fade):
    """
    تطبيق تأثير الصورة الشخصية (تظليل الخلفية وإضاءة الشخص).
    Apply portrait photo effect: darken background, keep person bright.

    منطق التأثير / Effect Logic:
    - fade = 0.0 → وضع سريع: عرض عادي مشرق بدون تغميق / Fast mode: normal bright view
    - fade = 1.0 → وضع بطيء: تأثير studio portrait: الشخص مضاء، الخلفية سوداء / Slow mode: studio portrait - person bright, background black
    """
    if segmentation_mask is None or fade < 0.01:
        return frame

    try:
        # تطبيع قناع التجزئة / Normalize segmentation mask
        mask = np.clip(segmentation_mask, 0, 1).astype(np.float32)

        # إعادة تحديد حجم القناع إذا لزم الأمر / Resize mask if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # تطبيق تمويه خفيف للقناع للحصول على انتقال سلس / Apply slight blur for smooth transition
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.clip(mask, 0, 1).astype(np.float32)

        # إنشاء صورة مظلمة جداً للخلفية / Create very dark background version
        dark_frame = cv2.multiply(frame.astype(np.float32), 0.12)

        # تحويل الإطار إلى float / Convert frame to float
        frame_float = frame.astype(np.float32)

        # دمج القناع إلى 3 قنوات / Stack mask into 3 channels
        mask_3ch = np.dstack([mask, mask, mask])

        # إنشاء الإطار مع التأثير الشخصي
        # Create portrait effect frame: bright person + dark background
        portrait_frame = frame_float * mask_3ch + dark_frame * (1 - mask_3ch)

        # دمج سلس: عندما fade=0 نعرض الإطار الطبيعي، عندما fade=1 نعرض التأثير الشخصي
        # Smooth transition: fade=0 → normal frame, fade=1 → portrait frame
        result = frame_float * (1.0 - fade) + portrait_frame * fade

        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception:
        # في حالة الخطأ، أرجع الإطار الأصلي / On error, return original frame
        return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # قلب الصورة أفقياً لتجربة مرآة طبيعية / Mirror-like user experience
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    current_time = time.time()
    speed = 0.0
    center = None
    move_dx, move_dy = 0.0, 0.0
    segmentation_mask = None

    # ================================
    # كشف الجسم / Full-body pose detect
    # ================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        # رسم الهيكل العظمي / Draw body skeleton landmarks
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 220, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 180, 80), thickness=2),
        )

        lms = result.pose_landmarks.landmark

        # الحصول على قناع التجزئة إن وجد / Get segmentation mask if available
        if result.segmentation_mask is not None:
            segmentation_mask = result.segmentation_mask

        left_hip = lms[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = lms[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # التحقق من الرؤية / Visibility check for stable center
        if left_hip.visibility > 0.35 and right_hip.visibility > 0.35:
            cx = int(((left_hip.x + right_hip.x) * 0.5) * frame_w)
            cy = int(((left_hip.y + right_hip.y) * 0.5) * frame_h)
            center = (cx, cy)

            # علامة على مركز الجسم / Visual body center marker
            cv2.circle(frame, center, 5, (0, 255, 255), -
                       1, lineType=cv2.LINE_AA)

            # حساب السرعة الحقيقية px/s / Real speed in pixels per second
            if prev_center is not None and prev_time is not None:
                dt = current_time - prev_time
                if dt > 1e-4:
                    move_dx = cx - prev_center[0]
                    move_dy = cy - prev_center[1]
                    dist = float(np.hypot(move_dx, move_dy))
                    speed = dist / dt

                    # إنشاء نجوم من الجسم الكامل إذا كانت السرعة عالية / Spawn stars from full body if fast
                    if speed > SPEED_THRESHOLD:
                        spawn_star_particles_from_body(
                            lms, frame_w, frame_h, move_dx, move_dy, speed)

            prev_center = center
            prev_time = current_time
        else:
            # إذا فقدنا تتبع الورك، لا نحسب سرعة / If hips are not visible, skip speed update
            prev_center = None
            prev_time = None
    else:
        # لا يوجد شخص مكتشف / No person detected
        prev_center = None
        prev_time = None

    # =====================================
    # منطق الصوت / Sound edge-trigger logic
    # =====================================
    if speed > SPEED_THRESHOLD:
        if not high_speed_active:
            high_speed_active = True
            if speed_sound is not None:
                sound_channel = speed_sound.play(loops=0)
    else:
        if high_speed_active:
            high_speed_active = False
            if sound_channel is not None:
                sound_channel.stop()

    # ==========================================
    # منطق انتقال الأوضاع (سريع/بطيء) / Mode transition logic
    # ==========================================
    # وضع سريع (السرعة > عتبة): portrait_fade → 0 (لا يوجد تأثير داكن) / Fast mode: portrait_fade → 0 (normal bright view)
    # وضع بطيء (السرعة ≤ عتبة): portrait_fade → 1 (تأثير studio portrait كامل) / Slow mode: portrait_fade → 1 (full dark background effect)
    if speed > SPEED_THRESHOLD:
        # وضع سريع: تقليل تأثير العمق تدريجياً / Fast mode: reduce portrait effect (toward 0)
        portrait_fade = max(0.0, portrait_fade - 0.08)
    else:
        # وضع بطيء: زيادة تأثير العمق تدريجياً / Slow mode: increase portrait effect (toward 1)
        portrait_fade = min(1.0, portrait_fade + 0.08)

    # ==========================================
    # تطبيق تأثير الصورة الشخصية / Apply portrait mode
    # ==========================================
    if portrait_fade > 0.01:
        frame = apply_portrait_effect(frame, segmentation_mask, portrait_fade)

    # =====================================
    # تحديث ورسم النجوم / Update and draw stars
    # =====================================
    update_and_draw_particles(frame)

    # ====================
    # حساب FPS / FPS meter
    # ====================
    dt_fps = current_time - fps_prev_time
    if dt_fps > 1e-6:
        fps_value = 1.0 / dt_fps
    fps_prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps_value:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Speed: {speed:.1f} px/s  Threshold: {SPEED_THRESHOLD:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # عرض حالة الوضع / Display current mode
    mode_text = "FAST MODE - STARS" if speed > SPEED_THRESHOLD else "SLOW MODE - PORTRAIT"
    mode_color = (0, 255, 100) if speed > SPEED_THRESHOLD else (100, 100, 255)
    cv2.putText(
        frame,
        mode_text,
        (10, frame_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_TITLE, frame)

    # خروج عند الضغط على q / Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =======================================
# تنظيف الموارد / Clean up resources at end
# =======================================
cap.release()
cv2.destroyAllWindows()
pose.close()
pygame.mixer.quit()
