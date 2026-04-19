import time
import random
import threading
from typing import Dict, List, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit.components.v1 as components


# ==============================================
# Streamlit page settings
# ==============================================

st.set_page_config(
    page_title="Creative Flow • Creative Block",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-0: #04050a;
        --bg-1: #09111f;
        --bg-2: #0f1a2e;
        --card: rgba(12, 18, 34, 0.72);
        --card-strong: rgba(12, 18, 34, 0.94);
        --border: rgba(125, 211, 252, 0.18);
        --text: #e8eefc;
        --muted: #92a1c4;
        --cyan: #38d9ff;
        --cyan-2: #74f3ff;
        --violet: #a78bfa;
        --magenta: #ff6bd6;
        --green: #5cffb0;
        --amber: #ffc766;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
        --glow-cyan: 0 0 24px rgba(56, 217, 255, 0.22);
        --glow-violet: 0 0 24px rgba(167, 139, 250, 0.18);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 217, 255, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 24%),
            radial-gradient(circle at 50% 0%, rgba(255, 107, 214, 0.06), transparent 30%),
            linear-gradient(180deg, #03040a 0%, #070b14 35%, #05070d 100%);
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, rgba(0,0,0,0.45), rgba(0,0,0,0.1));
        opacity: 0.25;
    }

    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            radial-gradient(circle, rgba(180, 245, 255, 0.22) 0.8px, transparent 1px),
            radial-gradient(circle, rgba(255, 178, 245, 0.14) 0.7px, transparent 1px);
        background-size: 120px 120px, 170px 170px;
        background-position: 0 0, 40px 65px;
        opacity: 0.24;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 12, 22, 0.92), rgba(7, 11, 19, 0.96));
        border-right: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 18px 0 60px rgba(0, 0, 0, 0.28);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .hero-wrap {
        padding: 0.25rem 0 0.7rem 0;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(125, 211, 252, 0.16);
        background: linear-gradient(135deg, rgba(14, 20, 36, 0.9), rgba(8, 12, 22, 0.8));
        border-radius: 28px;
        padding: 1.35rem 1.5rem;
        box-shadow: var(--shadow), var(--glow-cyan);
    }

    .hero-card::before {
        content: "";
        position: absolute;
        inset: -2px;
        background: linear-gradient(120deg, rgba(56, 217, 255, 0.18), transparent 30%, transparent 70%, rgba(167, 139, 250, 0.16));
        opacity: 0.7;
        pointer-events: none;
    }

    .hero-card > * { position: relative; z-index: 1; }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(56, 217, 255, 0.18);
        background: rgba(255, 255, 255, 0.03);
        color: var(--cyan-2);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: var(--glow-cyan);
    }

    .hero-title {
        margin-top: 0.8rem;
        font-size: clamp(2.2rem, 4vw, 3.85rem);
        line-height: 0.96;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: var(--text);
        text-shadow: 0 0 22px rgba(56, 217, 255, 0.12);
    }

    .hero-title span {
        background: linear-gradient(90deg, #ffffff 0%, #c7f5ff 22%, #7df0ff 46%, #b6a3ff 72%, #ff8ce6 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        margin-top: 0.8rem;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 56rem;
    }

    .hero-description {
        margin-top: 0.7rem;
        color: #b8c5e3;
        font-size: 0.95rem;
        line-height: 1.7;
        max-width: 62rem;
        border-left: 2px solid rgba(56, 217, 255, 0.35);
        padding-left: 0.9rem;
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.1rem 0 0.75rem;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #9eb4df;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(11, 16, 29, 0.86), rgba(8, 12, 21, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(14px);
    }

    .hud-bar {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.65rem 0 1rem 0;
    }

    .hud-chip {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, rgba(16, 23, 40, 0.9), rgba(8, 12, 21, 0.95));
        padding: 0.95rem 1rem;
        min-height: 78px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .hud-chip::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.05) 18%, transparent 38%);
        opacity: 0.5;
        pointer-events: none;
    }

    .hud-label {
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8aa0c8;
        font-weight: 700;
    }

    .hud-value {
        margin-top: 0.38rem;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text);
    }

    .hud-sub {
        margin-top: 0.18rem;
        font-size: 0.88rem;
        color: var(--muted);
    }

    .hud-fast { border-color: rgba(92, 255, 176, 0.24); box-shadow: 0 0 28px rgba(92, 255, 176, 0.1); }
    .hud-slow { border-color: rgba(255, 107, 214, 0.22); box-shadow: 0 0 28px rgba(255, 107, 214, 0.08); }
    .hud-cyan { border-color: rgba(56, 217, 255, 0.22); box-shadow: 0 0 28px rgba(56, 217, 255, 0.08); }

    .video-shell {
        position: relative;
        margin: 0.2rem auto 0;
        padding: 1rem;
        border-radius: 28px;
        max-width: 1120px;
        background: linear-gradient(180deg, rgba(10, 15, 27, 0.9), rgba(6, 10, 18, 0.98));
        border: 1px solid rgba(125, 211, 252, 0.14);
        box-shadow: var(--shadow), 0 0 0 1px rgba(255,255,255,0.02) inset, var(--glow-violet);
    }

    .video-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 28px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(56,217,255,0.35), rgba(167,139,250,0.22), rgba(255,107,214,0.2));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
        opacity: 0.45;
    }

    .video-hint {
        margin-top: 0.8rem;
        color: var(--muted);
        font-size: 0.92rem;
        text-align: center;
    }

    .stApp video,
    .stApp iframe,
    .stApp canvas {
        border-radius: 22px !important;
        border: 1px solid rgba(116, 243, 255, 0.4) !important;
        box-shadow:
            0 22px 70px rgba(0, 0, 0, 0.45),
            0 0 0 1px rgba(56, 217, 255, 0.22),
            0 0 18px rgba(56, 217, 255, 0.28),
            0 0 34px rgba(255, 107, 214, 0.20) !important;
        overflow: hidden !important;
    }

    .stApp video {
        display: block !important;
        width: min(560px, 78vw) !important;
        max-width: 560px !important;
        min-width: 0 !important;
        height: auto !important;
        margin: 0 auto !important;
        border: 1px solid rgba(125, 211, 252, 0.55) !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.05),
            0 0 28px rgba(56, 217, 255, 0.35),
            0 0 52px rgba(255, 107, 214, 0.22),
            0 26px 80px rgba(0, 0, 0, 0.5) !important;
    }

    .stApp .stVideo,
    .stApp [data-testid="stVideo"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    .camera-stage {
        margin: 0.15rem auto 0.05rem;
        text-align: center;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0,1fr));
        gap: 0.65rem;
        margin: 0.05rem auto 0.55rem auto;
        max-width: 620px;
    }

    .stats-card {
        background: linear-gradient(180deg, rgba(16,23,40,.9), rgba(8,12,21,.95));
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
    }

    @media (max-width: 860px) {
        .stats-grid { grid-template-columns: 1fr; }
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(14, 20, 36, 0.88), rgba(8, 12, 20, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stMetricLabel"] {
        color: #9eb4df !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em;
    }

    [data-testid="stMetricDelta"] {
        color: var(--muted) !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.4rem;
    }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #7b8eb7;
    }

    .stSlider [role="slider"] {
        box-shadow: 0 0 0 4px rgba(56, 217, 255, 0.08);
    }

    .stToggle [data-baseweb="toggle"] div {
        background: rgba(255,255,255,0.1);
    }

    .stMarkdown, .stText, p, li {
        color: var(--text);
    }

    .footer-note {
        margin-top: 1rem;
        text-align: center;
        color: #7f91b4;
        font-size: 0.88rem;
    }

    .credits-wrap {
        margin-top: 1.6rem;
        text-align: center;
        padding: 1.05rem 0.85rem;
        border-top: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, rgba(9, 13, 24, 0.2), rgba(7, 10, 18, 0.45));
    }

    .credits-text {
        color: #8fa2c9;
        font-size: 0.76rem;
        line-height: 1.8;
        letter-spacing: 0.02em;
    }

    @media (max-width: 768px) {
        .hud-bar { grid-template-columns: 1fr; }
        .hero-card { padding: 1.1rem; }
        .video-shell { padding: 0.7rem; border-radius: 22px; }
        .stApp video { width: min(92vw, 520px) !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-card">
                <div class="hero-kicker">✨ New Media Interactive Installation</div>
                <div class="hero-title"><span>Creative Flow • Creative Block</span></div>
                <div class="hero-subtitle">Live body movement as a metaphor for creative energy</div>
                <div class="hero-description">This experience visualizes the fluctuation of creative energy. Fast movement represents Creative Flow (stars trail), while stillness represents Creative Block (cinematic darkness).</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Real-time Video Processor (WebRTC)
# ============================================================
class MotionEffectsProcessor:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Runtime settings from the sidebar
        self.settings = {
            "speed_threshold": 45.0,
            "stars_min": 10,
            "stars_max": 15,
            "portrait_darkness": 0.12,
            "fade_step": 0.08,
            "sound_enabled": True,
            "draw_pose": True,
        }

        # Motion + particles state
        self.prev_center = None
        self.prev_time = None
        self.portrait_fade = 0.0
        self.high_speed_active = False
        self.particles: List[Dict] = []

        # Runtime stats
        self.speed = 0.0
        self.fps = 0.0
        self.last_frame_time = time.time()
        # Increments when transitioning into fast mode
        self.sound_event_count = 0

        # Lazy initialization - MediaPipe loads on first frame
        self.mp_pose = None
        self.mp_drawing = None
        self.pose = None
        self.body_landmarks = None
        self._initialized = False
        self.frame_counter = 0
        self.process_every_n_frames = 2
        self.prev_gray = None
        self.fallback_points = []

    def update_settings(self, new_settings: Dict) -> None:
        with self.lock:
            self.settings.update(new_settings)

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe on first frame to avoid blocking WebRTC"""
        if self._initialized:
            return

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                enable_segmentation=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

            self.body_landmarks = [
                self.mp_pose.PoseLandmark.NOSE.value,
                self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
                self.mp_pose.PoseLandmark.LEFT_EYE.value,
                self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
                self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
                self.mp_pose.PoseLandmark.RIGHT_EYE.value,
                self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
                self.mp_pose.PoseLandmark.LEFT_EAR.value,
                self.mp_pose.PoseLandmark.RIGHT_EAR.value,
                self.mp_pose.PoseLandmark.MOUTH_LEFT.value,
                self.mp_pose.PoseLandmark.MOUTH_RIGHT.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                self.mp_pose.PoseLandmark.LEFT_HEEL.value,
                self.mp_pose.PoseLandmark.RIGHT_HEEL.value,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            ]
        except Exception:
            # Keep stream alive even if model init fails in cloud environment.
            self.pose = None
            self.body_landmarks = []
        finally:
            self._initialized = True

    def get_runtime_stats(self) -> Dict:
        with self.lock:
            mode = "FAST" if self.speed > self.settings["speed_threshold"] else "SLOW"
            return {
                "speed": float(self.speed),
                "fps": float(self.fps),
                "mode": mode,
                "sound_event_count": int(self.sound_event_count),
            }

    # =========================================
    # Draw realistic 5-pointed star
    # =========================================
    @staticmethod
    def _star_points(x: int, y: int, outer_r: int, inner_r: int, angle_deg: float = 0.0) -> np.ndarray:
        pts = []
        start_angle = np.deg2rad(angle_deg - 90.0)
        for i in range(10):
            r = outer_r if i % 2 == 0 else inner_r
            theta = start_angle + i * (np.pi / 5.0)
            px = int(x + r * np.cos(theta))
            py = int(y + r * np.sin(theta))
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def draw_star(self, frame: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], angle: float = 0.0, alpha: float = 1.0) -> None:
        if alpha <= 0:
            return

        h, w = frame.shape[:2]
        if x < -40 or x > w + 40 or y < -40 or y > h + 40:
            return

        outer_r = max(2, int(size))
        inner_r = max(1, int(outer_r * 0.45))

        pts_main = self._star_points(x, y, outer_r, inner_r, angle)
        pts_glow = self._star_points(
            x, y, int(outer_r * 1.35), int(inner_r * 1.35), angle)

        glow_overlay = frame.copy()
        cv2.fillPoly(glow_overlay, [pts_glow],
                     (255, 255, 255), lineType=cv2.LINE_AA)
        glow_strength = 0.18 * float(alpha)
        cv2.addWeighted(glow_overlay, glow_strength, frame,
                        1.0 - glow_strength, 0, frame)

        star_overlay = frame.copy()
        cv2.fillPoly(star_overlay, [pts_main], color, lineType=cv2.LINE_AA)
        main_strength = max(0.05, min(1.0, float(alpha)))
        cv2.addWeighted(star_overlay, main_strength, frame,
                        1.0 - main_strength, 0, frame)

    # ====================================================
    # Full-body star particles
    # ====================================================
    def spawn_star_particles_from_body(self, lms, frame_w: int, frame_h: int, move_dx: float, move_dy: float, speed: float) -> None:
        mag = float(np.hypot(move_dx, move_dy))
        if mag < 1e-6:
            return

        with self.lock:
            stars_min = int(self.settings["stars_min"])
            stars_max = int(self.settings["stars_max"])

        dir_x = move_dx / mag
        dir_y = move_dy / mag
        trail_speed = float(np.clip(speed / 120.0, 1.2, 6.0))

        total_stars = random.randint(stars_min, stars_max)

        # Build a wider set of emit points so stars come from the whole body.
        visible_points = []
        for lm_idx in self.body_landmarks:
            if lm_idx < len(lms) and lms[lm_idx].visibility > 0.35:
                lm = lms[lm_idx]
                visible_points.append(
                    (int(lm.x * frame_w), int(lm.y * frame_h), lm.visibility))

        # Synthetic anchors between major body joints to fill gaps.
        anchor_pairs = [
            (self.mp_pose.PoseLandmark.NOSE.value,
             self.mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.NOSE.value,
             self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
             self.mp_pose.PoseLandmark.LEFT_HIP.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
             self.mp_pose.PoseLandmark.RIGHT_HIP.value),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value,
             self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value,
             self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value,
             self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
             self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
             self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value,
             self.mp_pose.PoseLandmark.RIGHT_HIP.value),
        ]

        for a_idx, b_idx in anchor_pairs:
            if a_idx < len(lms) and b_idx < len(lms):
                a = lms[a_idx]
                b = lms[b_idx]
                if a.visibility > 0.35 and b.visibility > 0.35:
                    for t in (0.15, 0.35, 0.5, 0.65, 0.85):
                        x = int(((a.x * (1 - t)) + (b.x * t)) * frame_w)
                        y = int(((a.y * (1 - t)) + (b.y * t)) * frame_h)
                        visible_points.append(
                            (x, y, min(a.visibility, b.visibility)))

        if not visible_points:
            return

        # Add a handful of back-trail emit points behind the movement direction.
        trail_x = -dir_x * random.uniform(12, 34)
        trail_y = -dir_y * random.uniform(12, 34)

        for _ in range(total_stars):
            px_orig, py_orig, vis = random.choice(visible_points)

            # Back trail offset + random spread
            offset_dist = random.uniform(15, 40)
            px = px_orig + trail_x + \
                (-dir_x * offset_dist) + random.uniform(-12, 12)
            py = py_orig + trail_y + \
                (-dir_y * offset_dist) + random.uniform(-12, 12)

            vx = -dir_x * trail_speed + random.uniform(-0.7, 0.7)
            vy = -dir_y * trail_speed + random.uniform(-0.7, 0.7)

            life = random.randint(20, 40)
            size = random.randint(8, 18)

            # Bright yellow-white (BGR)
            b = random.randint(170, 245)
            g = random.randint(225, 255)
            r = random.randint(235, 255)

            self.particles.append(
                {
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
                }
            )

    def update_and_draw_particles(self, frame: np.ndarray) -> None:
        now = time.time()
        alive = []

        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["angle"] += p["spin"]
            p["life"] -= 1

            if p["life"] <= 0:
                continue

            alpha = p["life"] / p["max_life"]
            tw = 0.75 + 0.25 * np.sin(now * 12.0 + p["twinkle"])
            draw_size = max(2, int(p["size"] * tw))

            self.draw_star(
                frame,
                int(p["x"]),
                int(p["y"]),
                draw_size,
                p["color"],
                angle=p["angle"],
                alpha=alpha,
            )
            alive.append(p)

        self.particles = alive

    def spawn_star_particles_generic(self, frame_w: int, frame_h: int, speed: float) -> None:
        """Fallback stars when body landmarks are unavailable."""
        with self.lock:
            stars_min = int(self.settings["stars_min"])
            stars_max = int(self.settings["stars_max"])

        total_stars = random.randint(max(4, stars_min // 2), max(8, stars_max))
        spread = int(np.clip(speed * 0.8, 30, 180))
        base_points = self.fallback_points if self.fallback_points else [
            (frame_w // 2, frame_h // 2)]

        for _ in range(total_stars):
            bx, by = random.choice(base_points)
            px = bx + random.randint(-spread, spread)
            py = by + random.randint(-spread, spread)
            vx = random.uniform(-2.2, 2.2)
            vy = random.uniform(-2.2, 2.2)
            life = random.randint(16, 34)
            size = random.randint(6, 14)

            b = random.randint(170, 245)
            g = random.randint(225, 255)
            r = random.randint(235, 255)

            self.particles.append(
                {
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
                }
            )

    def apply_global_darkening(self, frame: np.ndarray, fade: float) -> np.ndarray:
        if fade < 0.01:
            return frame
        with self.lock:
            darkness = float(self.settings["portrait_darkness"])
        frame_float = frame.astype(np.float32)
        dark_frame = cv2.multiply(frame_float, darkness)
        out = frame_float * (1.0 - fade) + dark_frame * fade
        return np.clip(out, 0, 255).astype(np.uint8)

    def _fallback_motion_speed(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.fallback_points = []
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        ys, xs = np.where(thresh > 0)
        if len(xs) > 0:
            sample_count = min(180, len(xs))
            idx = np.linspace(0, len(xs) - 1, sample_count, dtype=int)
            self.fallback_points = [(int(xs[i]), int(ys[i])) for i in idx]
        else:
            self.fallback_points = []

        return float(np.mean(diff)) * 8.0

    # ======================================================
    # Portrait dark-background effect
    # ======================================================
    def apply_portrait_effect(self, frame: np.ndarray, segmentation_mask: np.ndarray, fade: float) -> np.ndarray:
        if segmentation_mask is None or fade < 0.01:
            return frame

        try:
            mask = np.clip(segmentation_mask, 0, 1).astype(np.float32)

            # Ensure same dimensions as frame
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = np.clip(mask, 0, 1).astype(np.float32)

            with self.lock:
                darkness = float(self.settings["portrait_darkness"])

            frame_float = frame.astype(np.float32)
            dark_frame = cv2.multiply(frame_float, darkness)
            mask_3ch = np.dstack([mask, mask, mask])

            # Keep person bright, darken background
            portrait_frame = frame_float * mask_3ch + \
                dark_frame * (1.0 - mask_3ch)

            # Smooth transition by fade
            result = frame_float * (1.0 - fade) + portrait_frame * fade
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception:
            return frame

    # ==================================
    # Frame processing
    # ==================================
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        now = time.time()
        dt_fps = now - self.last_frame_time
        if dt_fps > 1e-6:
            self.fps = 1.0 / dt_fps
        self.last_frame_time = now

        self.frame_counter += 1

        try:
            # Initialize MediaPipe on first frame
            if not self._initialized:
                self._init_mediapipe()
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # If model is unavailable, keep raw stream alive.
            if self.pose is None:
                speed = self._fallback_motion_speed(img)
                with self.lock:
                    threshold = float(self.settings["speed_threshold"])
                    fade_step = float(self.settings["fade_step"])
                    sound_enabled = bool(self.settings["sound_enabled"])

                self.speed = speed
                if speed > threshold * 0.75:
                    self.spawn_star_particles_generic(w, h, speed)
                    self.portrait_fade = max(
                        0.0, self.portrait_fade - fade_step)
                    if not self.high_speed_active:
                        self.high_speed_active = True
                        if sound_enabled:
                            self.sound_event_count += 1
                else:
                    self.high_speed_active = False
                    self.portrait_fade = min(
                        1.0, self.portrait_fade + fade_step)

                img = self.apply_global_darkening(img, self.portrait_fade)
                self.update_and_draw_particles(img)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Reduce CPU load in cloud by processing every Nth frame.
            if self.frame_counter % self.process_every_n_frames != 0:
                self.update_and_draw_particles(img)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            speed = 0.0
            move_dx, move_dy = 0.0, 0.0
            segmentation_mask = None

            # Process on a smaller image for faster inference.
            proc_w = max(320, int(w * 0.6))
            proc_h = max(240, int(h * 0.6))
            proc_img = cv2.resize(img, (proc_w, proc_h),
                                  interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                if result.segmentation_mask is not None:
                    segmentation_mask = result.segmentation_mask

                # Body center using left and right hips
                left_hip = lms[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = lms[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

                if left_hip.visibility > 0.35 and right_hip.visibility > 0.35:
                    cx = int(((left_hip.x + right_hip.x) * 0.5) * w)
                    cy = int(((left_hip.y + right_hip.y) * 0.5) * h)
                    center = (cx, cy)

                    if self.prev_center is not None and self.prev_time is not None:
                        dt = now - self.prev_time
                        if dt > 1e-4:
                            move_dx = cx - self.prev_center[0]
                            move_dy = cy - self.prev_center[1]
                            dist = float(np.hypot(move_dx, move_dy))
                            speed = dist / dt

                            with self.lock:
                                threshold = float(
                                    self.settings["speed_threshold"])
                                sound_enabled = bool(
                                    self.settings["sound_enabled"])

                            # FAST: stars from full body + no portrait darkening
                            if speed > threshold:
                                self.spawn_star_particles_from_body(
                                    lms, w, h, move_dx, move_dy, speed)
                                if not self.high_speed_active:
                                    self.high_speed_active = True
                                    if sound_enabled:
                                        self.sound_event_count += 1
                            else:
                                self.high_speed_active = False

                    self.prev_center = center
                    self.prev_time = now

                    cv2.circle(img, center, 5, (0, 255, 255), -
                               1, lineType=cv2.LINE_AA)
                else:
                    self.prev_center = None
                    self.prev_time = None

                with self.lock:
                    draw_pose = bool(self.settings["draw_pose"])

                if draw_pose:
                    self.mp_drawing.draw_landmarks(
                        img,
                        result.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(80, 220, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(255, 180, 80), thickness=2),
                    )
            else:
                self.prev_center = None
                self.prev_time = None
                self.high_speed_active = False
                speed = self._fallback_motion_speed(img)
                if speed > 0:
                    with self.lock:
                        threshold = float(self.settings["speed_threshold"])
                        fade_step = float(self.settings["fade_step"])
                    if speed > threshold * 0.75:
                        self.spawn_star_particles_generic(w, h, speed)
                        self.portrait_fade = max(
                            0.0, self.portrait_fade - fade_step)
                    else:
                        self.portrait_fade = min(
                            1.0, self.portrait_fade + fade_step)

            with self.lock:
                threshold = float(self.settings["speed_threshold"])
                fade_step = float(self.settings["fade_step"])

            self.speed = speed

            if speed > threshold:
                self.portrait_fade = max(0.0, self.portrait_fade - fade_step)
            else:
                self.portrait_fade = min(1.0, self.portrait_fade + fade_step)

            if self.portrait_fade > 0.01:
                if segmentation_mask is not None:
                    img = self.apply_portrait_effect(
                        img, segmentation_mask, self.portrait_fade)
                else:
                    img = self.apply_global_darkening(img, self.portrait_fade)

            self.update_and_draw_particles(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            # Never break the stream loop because of a processing exception.
            self.prev_center = None
            self.prev_time = None
            self.high_speed_active = False
            self.speed = 0.0
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================
# Sidebar Controls
# ============================
with st.sidebar:

    st.markdown("""
    <style>
    .sidebar-label { font-size: 1.1rem; color: #0ff; font-weight: 600; margin-bottom: 4px; letter-spacing: 0.03em; }
    .sidebar-group { margin-bottom: 18px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group"><div class="sidebar-label">Flow Activation Threshold (px/s)</div></div>', unsafe_allow_html=True)
    speed_threshold = st.slider("Flow Activation Threshold (px/s)", 10.0, 200.0,
                                45.0, 1.0, help="Lower values trigger the fast star effect more easily.")

    st.markdown('<div class="sidebar-group"><div class="sidebar-label">Transition Smoothness</div></div>',
                unsafe_allow_html=True)
    fade_step = st.slider("Transition Smoothness", 0.02, 0.20, 0.08, 0.01,
                          help="Controls how quickly the app switches between fast and portrait states.")

    st.markdown('<div class="sidebar-group"><div class="sidebar-label">Star Density</div></div>',
                unsafe_allow_html=True)
    stars_range = st.slider("Star Density", 5, 30, (10, 15), 1,
                            help="Controls how many stars are emitted when fast movement is detected.")

    st.markdown('<div class="sidebar-group"><div class="sidebar-label">Block Darkness</div></div>',
                unsafe_allow_html=True)
    portrait_darkness = st.slider("Block Darkness", 0.05, 0.18, 0.12, 0.01,
                                  help="Makes the portrait mode background deeper and more dramatic.")

    st.markdown('<div class="sidebar-group"><div class="sidebar-label">Display</div></div>',
                unsafe_allow_html=True)
    draw_pose = st.toggle("Show skeleton", value=True,
                          help="Overlay the pose landmarks for debugging or visual styling.")
    sound_enabled = st.toggle(
        "Audio cue", value=True, help="Play a subtle in-browser beep when fast mode is entered.")

    st.markdown("<div class='footer-note'>Flow = stars • Block = darkness</div>",
                unsafe_allow_html=True)


# ========================================
# WebRTC Stream (Browser Camera Permission)
# ========================================
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ],
}

camera_left, camera_center, camera_right = st.columns([1.35, 2.3, 1.35])
with camera_center:
    st.markdown('<div class="camera-stage"></div>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="stars-trail-portrait",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 15, "max": 24},
                "facingMode": "user",
            },
            "audio": False,
        },
        video_processor_factory=MotionEffectsProcessor,
        async_processing=True,
    )


# Push sidebar settings to processor
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.update_settings(
        {
            "speed_threshold": speed_threshold,
            "stars_min": stars_range[0],
            "stars_max": stars_range[1],
            "portrait_darkness": portrait_darkness,
            "fade_step": fade_step,
            "sound_enabled": sound_enabled,
            "draw_pose": draw_pose,
        }
    )


# =========================================
# HUD + Browser sound trigger (best effort)
# =========================================
if "last_sound_event" not in st.session_state:
    st.session_state.last_sound_event = 0

hud_placeholder = st.empty()


def render_beep_once() -> None:
    # In-browser beep without external files
    components.html(
        """
        <script>
        (async () => {
          try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = 520;
            gain.gain.value = 0.001;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.03);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.28);
            osc.stop(ctx.currentTime + 0.30);
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )


def render_stats_and_sound_once() -> None:
    if not webrtc_ctx.video_processor:
        return

    stats = webrtc_ctx.video_processor.get_runtime_stats()

    # Play sound on each transition to FAST
    if sound_enabled and stats["sound_event_count"] > st.session_state.last_sound_event:
        st.session_state.last_sound_event = stats["sound_event_count"]
        render_beep_once()


# Refresh the HUD at a low cadence so status stays in sync with the camera.
if hasattr(st, "fragment"):
    @st.fragment(run_every="700ms")
    def live_fragment():
        render_stats_and_sound_once()

    live_fragment()
else:
    render_stats_and_sound_once()

st.markdown(
    """
    <div class="credits-wrap">
        <div class="credits-text">
            Hanin Mohamed Soliman &nbsp;•&nbsp; Student ID: 244861<br>
            Supervised by: Dr. Asmaa Mohamed<br>
            Assisting Staff: T.A. Rawan Hamada - T.A. Jana Khaled
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
