#!/usr/bin/env python3
# streamlit_gesture_music_autostart_clean.py

import os
import time
import math
import threading
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st

# optional imports
try:
    import mediapipe as mp
except Exception as e:
    st.error("mediapipe not installed. Install with: pip install mediapipe")
    raise SystemExit from e

# pygame may fail in some environments
AUDIO_AVAILABLE = True
try:
    import pygame
except Exception:
    AUDIO_AVAILABLE = False

# ---------------- CONFIG ----------------
PLAYLIST = ["song1.mp3", "song2.mp3", "song3.mp3"]  # replace with actual paths
CAM_INDEX_TRY = list(range(0, 4))
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FLIP_FRAME = True
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6
SMOOTH_WINDOW = 7
STABLE_REQUIRED = 5
COOLDOWN_SEC = 0.8
VOLUME_MIN_AREA = 0.03
VOLUME_MAX_AREA = 0.20

GEST_FIST = "fist"
GEST_OPEN = "open_palm"
GEST_POINT_R = "point_right"
GEST_POINT_L = "point_left"
GEST_VICTORY = "victory"
GEST_THUMBS_UP = "thumbs_up"
GEST_NONE = "none"

# ---------------- SHARED STATE ----------------
shared_state = {
    "frame": np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
    "gesture": GEST_NONE,
    "stable_gesture": GEST_NONE,
    "action": "—",
    "track": "—",
    "volume": 0.5,
    "camera_ok": False,
    "camera_index": None,
    "running": False,
    "error": None
}
state_lock = threading.Lock()

# ---------------- UTILITIES ----------------
def angle(a, b, c):
    ab, cb = a - b, c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-9
    cosang = np.clip(np.dot(ab, cb)/denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def bbox_from_landmarks(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))

def majority_vote(q):
    if not q: return GEST_NONE, 0
    c = Counter(q)
    return c.most_common(1)[0]

# ---------------- AUDIO ENGINE ----------------
class AudioEngine:
    def __init__(self, playlist):
        self.ok = AUDIO_AVAILABLE
        self.playlist = [p for p in playlist if os.path.exists(p)]
        self.idx = 0
        self.loaded = False
        self.paused = False
        if not self.ok: return
        try: pygame.mixer.init()
        except: self.ok = False; return
        if self.playlist: self.load_and_play(0)
    def load_and_play(self, i):
        if not (self.ok and self.playlist): return
        self.idx = i % len(self.playlist)
        try: pygame.mixer.music.load(self.playlist[self.idx]); pygame.mixer.music.play(); self.paused=False; self.loaded=True
        except: pass
    def play_or_resume(self):
        if not (self.ok and self.loaded): return
        if self.paused: pygame.mixer.music.unpause(); self.paused=False
    def pause(self):
        if not (self.ok and self.loaded): return
        pygame.mixer.music.pause(); self.paused=True
    def stop(self):
        if not (self.ok and self.loaded): return
        pygame.mixer.music.stop(); self.paused=False
    def next(self): self.load_and_play(self.idx+1)
    def prev(self): self.load_and_play(self.idx-1)
    def set_volume(self, v):
        if not (self.ok and self.loaded): return
        pygame.mixer.music.set_volume(float(np.clip(v,0.0,1.0)))
    def current_track_name(self):
        if not (self.ok and self.loaded): return "—"
        return os.path.basename(self.playlist[self.idx]) if self.playlist else "—"

# ---------------- GESTURE WORKER ----------------
class GestureWorker(threading.Thread):
    def __init__(self, audio_engine, camera_index=None):
        super().__init__(daemon=True)
        self.audio = audio_engine
        self.camera_index = camera_index
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, model_complexity=1,
                                         min_detection_confidence=MIN_DET_CONF,
                                         min_tracking_confidence=MIN_TRK_CONF)
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.stable_label = GEST_NONE
        self.stable_count = 0
        self.last_trigger_time = {g:0.0 for g in [GEST_FIST,GEST_OPEN,GEST_POINT_R,GEST_POINT_L,GEST_VICTORY,GEST_THUMBS_UP]}
    def to_np(self, pt): return np.array([pt.x, pt.y], dtype=np.float32)
    def finger_extended_angles(self,lms):
        P=[self.to_np(p) for p in lms]
        def is_extended(mcp,pip,dip,tip,thresh=160.0):
            ang = angle(P[mcp],P[pip],P[dip])
            ang2=angle(P[pip],P[dip],P[tip])
            return (ang>thresh) and (ang2>150.0)
        thumb_ext = angle(P[1],P[2],P[3])>150.0 and angle(P[2],P[3],P[4])>150.0
        return {"thumb":thumb_ext,"index":is_extended(5,6,7,8),"middle":is_extended(9,10,11,12),
                "ring":is_extended(13,14,15,16),"pinky":is_extended(17,18,19,20)}
    def classify_gesture(self,lms):
        f=self.finger_extended_angles(lms)
        if f["thumb"] and not any([f["index"],f["middle"],f["ring"],f["pinky"]]): return GEST_THUMBS_UP
        if not any(f.values()): return GEST_FIST
        if all(f.values()): return GEST_OPEN
        if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]: return GEST_VICTORY
        if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            idx_mcp=np.array([lms[5].x,lms[5].y],dtype=np.float32)
            idx_tip=np.array([lms[8].x,lms[8].y],dtype=np.float32)
            ang=math.degrees(math.atan2(idx_tip[1]-idx_mcp[1],idx_tip[0]-idx_mcp[0]))
            if -30<=ang<=30: return GEST_POINT_R
            if ang>=150 or ang<=-150: return GEST_POINT_L
        return GEST_NONE
    def map_volume_from_bbox_area(self,bbox):
        x1,y1,x2,y2=bbox
        w,h=max(0.0,x2-x1),max(0.0,y2-y1)
        area=w*h
        return float(np.clip((area-VOLUME_MIN_AREA)/(VOLUME_MAX_AREA-VOLUME_MIN_AREA+1e-9),0.0,1.0))
    def can_trigger(self,g):
        now=time.time()
        return self.stable_label==g and self.stable_count>=STABLE_REQUIRED and (now-self.last_trigger_time[g])>=COOLDOWN_SEC
    def run(self):
        cap=cv2.VideoCapture(self.camera_index if self.camera_index is not None else 0)
        if not cap.isOpened():
            with state_lock: shared_state["camera_ok"]=False; shared_state["error"]=f"Camera {self.camera_index} unavailable"; return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        with self.hands:
            while self.running:
                ret, frame = cap.read()
                if not ret: time.sleep(0.1); continue
                if FLIP_FRAME: frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=self.hands.process(rgb)
                label_now=GEST_NONE; volume=None
                if res.multi_hand_landmarks:
                    for h_lms in res.multi_hand_landmarks:
                        label_now=self.classify_gesture(h_lms.landmark)
                        bbox=bbox_from_landmarks(h_lms.landmark)
                        volume=self.map_volume_from_bbox_area(bbox)
                        try: self.mp_draw.draw_landmarks(frame,h_lms,self.mp_hands.HAND_CONNECTIONS)
                        except: pass
                self.pred_queue.append(label_now)
                maj,cnt=majority_vote(self.pred_queue)
                if maj==self.stable_label: self.stable_count=min(self.stable_count+1,SMOOTH_WINDOW)
                else: self.stable_label=maj; self.stable_count=cnt
                nowt=time.time(); action="—"
                if self.stable_label==GEST_OPEN and volume is not None:
                    if self.audio.ok and self.audio.loaded: self.audio.set_volume(volume)
                    action=f"Stop+Vol {int(volume*100)}%"
                    if self.can_trigger(GEST_OPEN) and self.audio.ok and self.audio.loaded: self.audio.stop(); self.last_trigger_time[GEST_OPEN]=nowt
                elif self.can_trigger(GEST_VICTORY): self.audio.play_or_resume(); action="Play ▶"; self.last_trigger_time[GEST_VICTORY]=nowt
                elif self.can_trigger(GEST_THUMBS_UP): self.audio.pause(); action="Pause ⏸"; self.last_trigger_time[GEST_THUMBS_UP]=nowt
                elif self.can_trigger(GEST_POINT_R): self.audio.next(); action="Next ▶▶"; self.last_trigger_time[GEST_POINT_R]=nowt
                elif self.can_trigger(GEST_POINT_L): self.audio.prev(); action="Prev ◀◀"; self.last_trigger_time[GEST_POINT_L]=nowt
                elif self.can_trigger(GEST_FIST): action="Fist (no-op)"
                with state_lock:
                    shared_state.update({"frame":cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                                         "gesture":label_now,"stable_gesture":self.stable_label,
                                         "action":action,"track":self.audio.current_track_name(),
                                         "volume":0.0 if volume is None else volume,"camera_ok":True,"error":None})
                time.sleep(0.02)
        cap.release()

# ---------------- CAMERA AUTO-DETECT ----------------
def find_working_camera(indices=CAM_INDEX_TRY):
    for idx in indices:
        cap=cv2.VideoCapture(idx)
        if not cap.isOpened(): cap.release(); continue
        ok,_=cap.read(); cap.release()
        if ok: return idx
    return None

# ---------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide", page_title="Gesture Music Player")
st.title("🎵 Gesture Music Player — Victory=Play, ThumbsUp=Pause")

# audio engine
if "audio" not in st.session_state: st.session_state.audio=AudioEngine(PLAYLIST)

# start worker automatically
if "worker" not in st.session_state or st.session_state.worker is None:
    cam_idx=find_working_camera()
    if cam_idx is None: st.error("No working camera detected.")
    else:
        shared_state["camera_index"]=cam_idx
        shared_state["running"]=True
        st.session_state.worker=GestureWorker(st.session_state.audio,camera_index=cam_idx)
        st.session_state.worker.start()

# sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.session_state.audio.ok and st.session_state.audio.loaded:
        vol=st.slider("Volume",0,100,int(shared_state["volume"]*100))
        st.session_state.audio.set_volume(vol/100)
    if st.button("Play (manual)"): st.session_state.audio.play_or_resume()
    if st.button("Pause (manual)"): st.session_state.audio.pause()
    if st.button("Next (manual)"): st.session_state.audio.next()
    if st.button("Prev (manual)"): st.session_state.audio.prev()

# main UI
col1,col2=st.columns([3,1])
with col1:
    st.markdown("### Camera feed")
    img_placeholder = st.empty()
with col2:
    st.markdown("### Status")
    status_placeholder = st.empty()

# ---------------- LIVE LOOP ----------------
while True:
    with state_lock:
        img_placeholder.image(shared_state["frame"],channels="RGB",use_container_width=True)
        status_placeholder.markdown(
            f"**Gesture (raw):** {shared_state['gesture']}  \n"
            f"**Gesture (stable):** {shared_state['stable_gesture']}  \n"
            f"**Action:** {shared_state['action']}  \n"
            f"**Track:** {shared_state['track']}  \n"
            f"**Volume:** {int(shared_state['volume']*100)}%  \n"
            f"**Camera OK:** {shared_state['camera_ok']}  \n"
            + (f"**Camera idx:** {shared_state['camera_index']}  \n" if shared_state.get("camera_index") is not None else "")
            + (f"**Error:** {shared_state['error']}" if shared_state.get("error") else "")
        )
    time.sleep(0.05)
