import time
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

import os
from deepface import DeepFace
from scipy.spatial import distance
from pymongo import MongoClient

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def load_env(path: str = ".env") -> None:
	file_path = Path(__file__).resolve().parent / path
	if not file_path.exists():
		return
	for line in file_path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		os.environ.setdefault(key.strip(), value.strip())

try:
	mp_pose = mp.solutions.pose
	mp_drawing = mp.solutions.drawing_utils
	POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
	USE_TASKS = False
except AttributeError:
	mp_pose = None
	mp_drawing = None
	POSE_CONNECTIONS = None
	USE_TASKS = True


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
	ba = a - b
	bc = c - b
	denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
	if denom == 0:
		return 0.0
	cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
	return float(np.degrees(np.arccos(cos_angle)))


def angle_3pt_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
	ba = a[:2] - b[:2]
	bc = c[:2] - b[:2]
	denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
	if denom == 0:
		return 0.0
	cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
	return float(np.degrees(np.arccos(cos_angle)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	return (a + b) / 2.0


@dataclass
class ExerciseState:
	reps: int = 0
	state: str = "down"
	last_side: Optional[str] = None
	last_transition_time: float = field(default_factory=time.time)
	hold_start: Optional[float] = None
	hold_seconds: float = 0.0
	ema_knee: Optional[float] = None
	stable_down_frames: int = 0
	stable_up_frames: int = 0

class FaceIDManager:
	def __init__(self, db_uri: Optional[str] = None):
		load_env()
		mongo_uri = db_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
		self.client = MongoClient(mongo_uri)
		self.db = self.client["Users"]
		self.patients_col = self.db["Galit_Tauber"]
		self.active_user = None
		self.known_patients = []
		self.last_error: Optional[str] = None
		self.match_threshold = float(os.getenv("FACEID_THRESHOLD", "0.6"))

	def load_db(self):
		"""Pulls face vectors from MongoDB into memory."""
		try:
			self.known_patients = list(self.patients_col.find({}, {"name": 1, "face_vector": 1}))
			print(f"Loaded {len(self.known_patients)} patients from Database.")
			if not self.known_patients:
				self.last_error = "No patients with face vectors found in DB. Run enroll_patients.py."
			else:
				self.last_error = None
		except Exception as exc:
			self.last_error = f"DB load failed: {exc}"
			print(self.last_error)

	def identify(self, frame):
		"""Compares webcam frame against the database."""
		try:
			# Generate vector for the live person
			results = DeepFace.represent(img_path=frame, model_name="Facenet512", enforce_detection=False)
			if not results:
				self.last_error = "No face detected in frame."
				return None
			live_vec = results[0]["embedding"]

			# Compare to stored vectors
			best_name = None
			best_dist = None
			for patient in self.known_patients:
				dist = distance.cosine(live_vec, patient["face_vector"])
				if best_dist is None or dist < best_dist:
					best_dist = dist
					best_name = patient.get("name")
			if best_dist is not None and best_dist < self.match_threshold:
				self.last_error = None
				return best_name
			if best_dist is None:
				self.last_error = "No face vectors to compare."
			else:
				self.last_error = (
					f"No match below threshold. Best={best_name} dist={best_dist:.3f} "
					f"(threshold={self.match_threshold:.2f})"
				)
		except Exception as exc:
			self.last_error = f"FaceID error: {exc}"
			return None
		return None

	def get_status(self) -> str:
		if self.active_user:
			return f"Active user: {self.active_user}"
		return self.last_error or "Scanning..."

	def log_workout(self, exercise, reps):
		"""Updates the specific user's workout log in MongoDB."""
		if self.active_user:
			self.db.workout_logs.update_one(
				{"patient_name": self.active_user, "exercise": exercise, "date": time.strftime("%Y-%m-%d")},
				{"$set": {"reps": reps, "last_sync": time.time()}},
				upsert=True
			)


class ElevenLabsSpeaker:
	def __init__(self) -> None:
		load_env()
		self.api_key = os.getenv("ELEVENLABS_API_KEY")
		self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
		self.model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
		self.min_gap_seconds = 1.0
		self.enabled = bool(self.api_key and self.voice_id)
		self.last_spoken: Dict[str, float] = {}
		self.queue: Queue[str] = Queue()
		self.worker = threading.Thread(target=self._run, daemon=True)
		if self.enabled:
			self.worker.start()
		else:
			print("[ElevenLabs] Disabled (missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID).")

	def say_feedback(self, exercise_key: str, text: str) -> None:
		if not self.enabled or not text:
			return
		now = time.time()
		last_time = self.last_spoken.get(exercise_key, 0.0)
		if now - last_time < self.min_gap_seconds:
			return
		self.last_spoken[exercise_key] = now
		self.queue.put(text)

	def _run(self) -> None:
		while True:
			try:
				text = self.queue.get()
			except Empty:
				continue
			try:
				pcm = self._synthesize_pcm(text)
				if pcm is None:
					continue
				audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
				sd.play(audio, samplerate=16000, blocking=True)
			except Exception as exc:
				print(f"[ElevenLabs] TTS error: {exc}")

	def _synthesize_pcm(self, text: str) -> Optional[bytes]:
		url = (
			f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
			"?output_format=pcm_16000"
		)
		payload = {
			"text": text,
			"model_id": self.model_id,
			"voice_settings": {
				"stability": 0.4,
				"similarity_boost": 0.75,
			},
		}
		data = json.dumps(payload).encode("utf-8")
		req = Request(
			url,
			data=data,
			method="POST",
			headers={
				"xi-api-key": self.api_key or "",
				"Content-Type": "application/json",
				"Accept": "audio/pcm",
			},
		)
		with urlopen(req, timeout=10) as resp:
			return resp.read()

class ExerciseDetector:
	def __init__(self) -> None:
		self.exercise: Optional[str] = None
		self.auto_mode = True
		self.min_confidence = 0.3
		self.current_confidence = 1.0
		self.lock_until = 0.0
		self.locked_exercise: Optional[str] = None
		self.allow_count = True
		self.allowed_exercises = {"squat", "lateral_lunge"}
		self.auto_lock_seconds = 3.0
		self.min_confidence_by_exercise = {
			"squat": 0.35,
			"lateral_lunge": 0.35,
		}
		self.states: Dict[str, ExerciseState] = {
			"squat": ExerciseState(),
			"lateral_lunge": ExerciseState(),
			"chest_press": ExerciseState(),
			"vertical_traction": ExerciseState(),
			"hip_raise": ExerciseState(),
			"airplane": ExerciseState(),
			"front_bridge": ExerciseState(),
			"alternate_leg_lowers": ExerciseState(),
			"thoracic_rotation": ExerciseState(),
		}

	def reset_exercises(self, names: Optional[set[str]] = None) -> None:
		reset_targets = names or set(self.states.keys())
		for name in reset_targets:
			st = self.states.get(name)
			if not st:
				continue
			st.reps = 0
			st.state = "down"
			st.last_side = None
			st.last_transition_time = time.time()
			st.hold_start = None
			st.hold_seconds = 0.0
			st.ema_knee = None
			st.stable_down_frames = 0
			st.stable_up_frames = 0

	def set_exercise(self, name: str) -> None:
		if name in self.allowed_exercises:
			self.exercise = name
			self.auto_mode = False
			self.locked_exercise = None
			self.lock_until = 0.0

	def set_auto(self, enabled: bool) -> None:
		self.auto_mode = enabled
		if enabled:
			self.exercise = None
			self.locked_exercise = None
			self.lock_until = 0.0

	def process(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		if self.auto_mode:
			f = self._features(lm)
			ankle_span = float(f["ankle_span"])
			lateral_shift = float(f["lateral_shift"])
			if self.locked_exercise == "lateral_lunge" and ankle_span < 0.18 and abs(lateral_shift) < 0.01:
				self.lock_until = 0.0
			now = time.time()
			if now < self.lock_until and self.locked_exercise in self.allowed_exercises:
				self.exercise = self.locked_exercise
				self.current_confidence = self._confidence(lm, self.exercise)
			else:
				exercise, conf = self._auto_detect(lm)
				self.exercise = exercise
				self.current_confidence = conf
				if conf >= self.min_confidence:
					self.locked_exercise = exercise
					self.lock_until = now + self.auto_lock_seconds
		else:
			if self.exercise not in self.allowed_exercises:
				self.exercise = "squat"
			self.current_confidence = self._confidence(lm, self.exercise)

		if not self.exercise:
			return "Detecting...", self.states["squat"], f"Conf: {self.current_confidence:.0%}"
		if self.exercise == "squat":
			ex_name, ex_state, status_text = self._detect_squat(lm)
		elif self.exercise == "lateral_lunge":
			ex_name, ex_state, status_text = self._detect_lateral_lunge(lm)
		elif self.exercise == "chest_press":
			ex_name, ex_state, status_text = self._detect_chest_press(lm)
		elif self.exercise == "vertical_traction":
			ex_name, ex_state, status_text = self._detect_vertical_traction(lm)
		elif self.exercise == "hip_raise":
			ex_name, ex_state, status_text = self._detect_hip_raise(lm)
		elif self.exercise == "airplane":
			ex_name, ex_state, status_text = self._detect_airplane(lm)
		elif self.exercise == "front_bridge":
			ex_name, ex_state, status_text = self._detect_front_bridge(lm)
		elif self.exercise == "alternate_leg_lowers":
			ex_name, ex_state, status_text = self._detect_alternate_leg_lowers(lm)
		elif self.exercise == "thoracic_rotation":
			ex_name, ex_state, status_text = self._detect_thoracic_rotation(lm)
		else:
			ex_name, ex_state, status_text = self.exercise, self.states[self.exercise], ""

		conf_text = f"Conf: {self.current_confidence:.0%}"
		if status_text:
			status_text = f"{status_text} | {conf_text}"
		else:
			status_text = conf_text
		return ex_name, ex_state, status_text

	def _features(self, lm: Dict[str, np.ndarray]) -> Dict[str, float | bool]:
		knee_l = angle_3pt_2d(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt_2d(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
		elbow_l = angle_3pt_2d(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt_2d(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		body_l = angle_3pt_2d(lm["shoulder_l"], lm["hip_l"], lm["ankle_l"])
		body_r = angle_3pt_2d(lm["shoulder_r"], lm["hip_r"], lm["ankle_r"])
		avg_knee = (knee_l + knee_r) / 2.0
		avg_elbow = (elbow_l + elbow_r) / 2.0
		avg_body = (body_l + body_r) / 2.0
		wrists_above = (lm["wrist_l"][1] < lm["shoulder_l"][1] and lm["wrist_r"][1] < lm["shoulder_r"][1])
		wrists_below = (lm["wrist_l"][1] > lm["shoulder_l"][1] and lm["wrist_r"][1] > lm["shoulder_r"][1])
		hip_center = midpoint(lm["hip_l"], lm["hip_r"])
		ankle_mid = midpoint(lm["ankle_l"], lm["ankle_r"])
		lateral_shift = hip_center[0] - ankle_mid[0]
		ankle_span = abs(lm["ankle_l"][0] - lm["ankle_r"][0])
		shoulder_span = abs(lm["shoulder_l"][0] - lm["shoulder_r"][0])
		hip_span = abs(lm["hip_l"][0] - lm["hip_r"][0])
		front_facing = shoulder_span > hip_span * 0.9
		knee_bent = avg_knee < 125
		lunge_left = knee_l < 135 and knee_r > 145 and (lateral_shift < -0.01 or ankle_span > 0.25)
		lunge_right = knee_r < 135 and knee_l > 145 and (lateral_shift > 0.01 or ankle_span > 0.25)
		elbows_bent = avg_elbow < 110
		body_straight = avg_body > 165
		elbows_ok = 70 < elbow_l < 110 and 70 < elbow_r < 110
		arms_out = elbow_l > 160 and elbow_r > 160
		return {
			"knee_l": knee_l,
			"knee_r": knee_r,
			"avg_knee": avg_knee,
			"elbow_l": elbow_l,
			"elbow_r": elbow_r,
			"avg_elbow": avg_elbow,
			"avg_body": avg_body,
			"wrists_above": wrists_above,
			"wrists_below": wrists_below,
			"lateral_shift": lateral_shift,
			"ankle_span": ankle_span,
			"shoulder_span": shoulder_span,
			"hip_span": hip_span,
			"front_facing": front_facing,
			"knee_bent": knee_bent,
			"lunge_left": lunge_left,
			"lunge_right": lunge_right,
			"elbows_bent": elbows_bent,
			"body_straight": body_straight,
			"elbows_ok": elbows_ok,
			"arms_out": arms_out,
		}

	def _clamp01(self, value: float) -> float:
		return max(0.0, min(1.0, value))

	def _confidence_from_features(self, f: Dict[str, float | bool], exercise: str) -> Optional[float]:
		knee_l = float(f["knee_l"])
		knee_r = float(f["knee_r"])
		avg_knee = float(f["avg_knee"])
		avg_elbow = float(f["avg_elbow"])
		avg_body = float(f["avg_body"])
		lateral_shift = float(f["lateral_shift"])
		ankle_span = float(f["ankle_span"])
		shoulder_span = float(f["shoulder_span"])
		hip_span = float(f["hip_span"])
		front_facing = bool(f["front_facing"])
		wrists_above = bool(f["wrists_above"])
		wrists_below = bool(f["wrists_below"])
		knee_bent = bool(f["knee_bent"])
		lunge_left = bool(f["lunge_left"])
		lunge_right = bool(f["lunge_right"])
		elbows_bent = bool(f["elbows_bent"])
		body_straight = bool(f["body_straight"])
		elbows_ok = bool(f["elbows_ok"])
		arms_out = bool(f["arms_out"])

		if exercise == "squat":
			score = 0.0
			if avg_knee < 140:
				score += 0.5
			if abs(knee_l - knee_r) < 20:
				score += 0.2
			if abs(lateral_shift) < 0.02 and ankle_span < 0.22:
				score += 0.2
			if avg_body > 140:
				score += 0.1
			if hip_span > 0 and (shoulder_span / hip_span) < 0.75:
				score += 0.1
			return self._clamp01(score)
		if exercise == "lateral_lunge":
			score = 0.0
			if lunge_left or lunge_right:
				score += 0.55
			if abs(lateral_shift) > 0.01:
				score += 0.2
			if ankle_span > 0.2:
				score += 0.2
			if abs(knee_l - knee_r) > 15:
				score += 0.05
			if front_facing:
				score += 0.1
			return self._clamp01(score)
		if exercise == "chest_press":
			score = 0.0
			if wrists_below:
				score += 0.3
			if elbows_bent:
				score += 0.4
			if avg_elbow < 100:
				score += 0.2
			if avg_body > 150:
				score += 0.1
			return self._clamp01(score)
		if exercise == "vertical_traction":
			score = 0.0
			if wrists_above:
				score += 0.4
			if avg_elbow > 150:
				score += 0.4
			if avg_elbow > 165:
				score += 0.2
			return self._clamp01(score)
		if exercise == "front_bridge":
			score = 0.0
			if body_straight:
				score += 0.5
			if elbows_ok:
				score += 0.4
			if avg_body > 170:
				score += 0.1
			return self._clamp01(score)
		if exercise == "airplane":
			score = 0.0
			if arms_out:
				score += 0.4
			if avg_body < 150:
				score += 0.4
			if avg_body < 140:
				score += 0.2
			return self._clamp01(score)
		if exercise == "alternate_leg_lowers":
			score = 0.0
			if (knee_l > 160 and knee_r < 120) or (knee_r > 160 and knee_l < 120):
				score += 0.8
			if abs(knee_l - knee_r) > 40:
				score += 0.2
			return self._clamp01(score)
		if exercise == "thoracic_rotation":
			return None
		if exercise == "hip_raise":
			return None
		return None

	def _confidence(self, lm: Dict[str, np.ndarray], exercise: str) -> float:
		f = self._features(lm)
		from_features = self._confidence_from_features(f, exercise)
		if from_features is not None:
			return from_features

		if exercise == "hip_raise":
			hip_angle_l = angle_3pt_2d(lm["shoulder_l"], lm["hip_l"], lm["knee_l"])
			hip_angle_r = angle_3pt_2d(lm["shoulder_r"], lm["hip_r"], lm["knee_r"])
			avg_hip = (hip_angle_l + hip_angle_r) / 2.0
			hip_y = (lm["hip_l"][1] + lm["hip_r"][1]) / 2.0
			knee_y = (lm["knee_l"][1] + lm["knee_r"][1]) / 2.0
			score = 0.0
			if avg_hip < 140 or hip_y > knee_y:
				score += 0.3
			if avg_hip > 160 and hip_y < knee_y:
				score += 0.5
			if abs(hip_y - knee_y) > 0.02:
				score += 0.2
			return self._clamp01(score)

		if exercise == "thoracic_rotation":
			shoulder_z_diff = lm["shoulder_l"][2] - lm["shoulder_r"][2]
			score = 0.0
			if abs(shoulder_z_diff) > 0.10:
				score += 0.6
			if abs(shoulder_z_diff) > 0.15:
				score += 0.2
			if abs(shoulder_z_diff) > 0.20:
				score += 0.2
			return self._clamp01(score)

		return 1.0

	def _auto_detect(self, lm: Dict[str, np.ndarray]) -> Tuple[str, float]:
		f = self._features(lm)
		ankle_span = float(f["ankle_span"])
		lateral_shift = float(f["lateral_shift"])
		front_facing = bool(f["front_facing"])
		if front_facing and (ankle_span > 0.24 and abs(lateral_shift) > 0.02):
			lunge_conf = self._confidence_from_features(f, "lateral_lunge") or 0.0
			return "lateral_lunge", lunge_conf
		if not front_facing and ankle_span < 0.2 and abs(lateral_shift) < 0.01:
			squat_conf = self._confidence_from_features(f, "squat") or 0.0
			return "squat", squat_conf
		if front_facing and ankle_span < 0.2 and abs(lateral_shift) < 0.015:
			squat_conf = self._confidence_from_features(f, "squat") or 0.0
			return "squat", squat_conf
		squat_conf = self._confidence_from_features(f, "squat") or 0.0
		lunge_conf = self._confidence_from_features(f, "lateral_lunge") or 0.0
		if lunge_conf > squat_conf:
			return "lateral_lunge", lunge_conf
		return "squat", squat_conf

	def _can_count(self) -> bool:
		if not self.allow_count:
			return False
		min_conf = self.min_confidence_by_exercise.get(self.exercise or "", self.min_confidence)
		return self.current_confidence >= min_conf

	def _transition(self, state: ExerciseState, new_state: str, now: float) -> None:
		if state.state != new_state:
			state.state = new_state
			state.last_transition_time = now

	def _detect_squat(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["squat"]
		knee_l = angle_3pt_2d(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt_2d(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
		body_l = angle_3pt_2d(lm["shoulder_l"], lm["hip_l"], lm["ankle_l"])
		body_r = angle_3pt_2d(lm["shoulder_r"], lm["hip_r"], lm["ankle_r"])
		avg_body = (body_l + body_r) / 2.0
		min_knee = min(knee_l, knee_r)
		if st.ema_knee is None:
			st.ema_knee = min_knee
		else:
			st.ema_knee = 0.3 * min_knee + 0.7 * st.ema_knee

		now = time.time()
		if st.ema_knee < 110:
			st.stable_down_frames += 1
			st.stable_up_frames = 0
		else:
			st.stable_down_frames = 0

		if st.ema_knee > 165:
			st.stable_up_frames += 1
			st.stable_down_frames = 0
		else:
			st.stable_up_frames = 0

		if st.stable_down_frames >= 3:
			self._transition(st, "down", now)
		if st.stable_up_frames >= 3 and st.state == "down":
			if self._can_count():
				st.reps += 1
			self._transition(st, "up", now)

		feedback = ""
		if st.state == "down":
			if st.ema_knee > 130:
				feedback = "Go lower"
			elif abs(knee_l - knee_r) > 30:
				feedback = "Keep knees even"
			elif avg_body < 145:
				feedback = "Keep your back straighter"
		status = f"Knee: {st.ema_knee:.0f}°"
		if feedback:
			status = f"{status} | {feedback}"
		return "Squats", st, status

	def _detect_lateral_lunge(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["lateral_lunge"]
		knee_l = angle_3pt_2d(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt_2d(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
		hip_center = midpoint(lm["hip_l"], lm["hip_r"])
		ankle_mid = midpoint(lm["ankle_l"], lm["ankle_r"])
		lateral_shift = hip_center[0] - ankle_mid[0]
		now = time.time()
		ankle_span = abs(lm["ankle_l"][0] - lm["ankle_r"][0])
		down_left = knee_l < 135 and knee_r > 145 and (lateral_shift < -0.01 or ankle_span > 0.25)
		down_right = knee_r < 135 and knee_l > 145 and (lateral_shift > 0.01 or ankle_span > 0.25)
		if down_left or down_right:
			self._transition(st, "down", now)
		if knee_l > 160 and knee_r > 160 and st.state == "down":
			if self._can_count():
				st.reps += 1
			self._transition(st, "up", now)
		side = "L" if down_left else "R" if down_right else ""
		feedback = ""
		if not down_left and not down_right:
			if ankle_span < 0.2 and abs(lateral_shift) < 0.01:
				feedback = "Step wider"
		elif down_left and knee_r < 150:
			feedback = "Straighten the right leg"
		elif down_right and knee_l < 150:
			feedback = "Straighten the left leg"
		status = f"Side: {side}"
		if feedback:
			status = f"{status} | {feedback}"
		return "Dynamic Hip Mobility Lateral Lunge", st, status

	def _detect_chest_press(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["chest_press"]
		elbow_l = angle_3pt(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		avg_elbow = (elbow_l + elbow_r) / 2.0
		now = time.time()
		if avg_elbow < 95:
			self._transition(st, "down", now)
		if avg_elbow > 160 and st.state == "down":
			if self._can_count():
				st.reps += 1
			self._transition(st, "up", now)
		return "Chest Press", st, f"Elbow: {avg_elbow:.0f}°"

	def _detect_vertical_traction(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["vertical_traction"]
		elbow_l = angle_3pt(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		wrists_below = (lm["wrist_l"][1] > lm["shoulder_l"][1] and lm["wrist_r"][1] > lm["shoulder_r"][1])
		wrists_above = (lm["wrist_l"][1] < lm["shoulder_l"][1] and lm["wrist_r"][1] < lm["shoulder_r"][1])
		avg_elbow = (elbow_l + elbow_r) / 2.0
		now = time.time()
		if avg_elbow < 100 and wrists_below:
			self._transition(st, "down", now)
		if avg_elbow > 160 and wrists_above and st.state == "down":
			if self._can_count():
				st.reps += 1
			self._transition(st, "up", now)
		return "Vertical Traction", st, f"Elbow: {avg_elbow:.0f}°"

	def _detect_hip_raise(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["hip_raise"]
		hip_angle_l = angle_3pt(lm["shoulder_l"], lm["hip_l"], lm["knee_l"])
		hip_angle_r = angle_3pt(lm["shoulder_r"], lm["hip_r"], lm["knee_r"])
		avg_hip = (hip_angle_l + hip_angle_r) / 2.0
		hip_y = (lm["hip_l"][1] + lm["hip_r"][1]) / 2.0
		knee_y = (lm["knee_l"][1] + lm["knee_r"][1]) / 2.0
		now = time.time()
		if avg_hip < 130 or hip_y > knee_y:
			self._transition(st, "down", now)
		if avg_hip > 160 and hip_y < knee_y and st.state == "down":
			if self._can_count():
				st.reps += 1
			self._transition(st, "up", now)
		return "Bent Knee Hip Raise", st, f"Hip: {avg_hip:.0f}°"

	def _detect_airplane(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["airplane"]
		hinge_l = angle_3pt(lm["shoulder_l"], lm["hip_l"], lm["ankle_l"])
		hinge_r = angle_3pt(lm["shoulder_r"], lm["hip_r"], lm["ankle_r"])
		hinge = (hinge_l + hinge_r) / 2.0
		elbow_l = angle_3pt(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		arms_out = elbow_l > 160 and elbow_r > 160
		now = time.time()
		holding = hinge < 145 and arms_out
		if holding:
			if st.hold_start is None:
				st.hold_start = now
			st.hold_seconds = now - st.hold_start
			st.state = "hold"
		else:
			if st.hold_start is not None and st.hold_seconds > 1.0:
				if self._can_count():
					st.reps += 1
			st.hold_start = None
			st.hold_seconds = 0.0
			st.state = "rest"
		return "Airplane Exercise", st, f"Hold: {st.hold_seconds:.1f}s"

	def _detect_front_bridge(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["front_bridge"]
		body_l = angle_3pt(lm["shoulder_l"], lm["hip_l"], lm["ankle_l"])
		body_r = angle_3pt(lm["shoulder_r"], lm["hip_r"], lm["ankle_r"])
		body = (body_l + body_r) / 2.0
		elbow_l = angle_3pt(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		elbows_ok = 70 < elbow_l < 110 and 70 < elbow_r < 110
		now = time.time()
		holding = body > 165 and elbows_ok
		if holding:
			if st.hold_start is None:
				st.hold_start = now
			st.hold_seconds = now - st.hold_start
			st.state = "hold"
		else:
			if st.hold_start is not None and st.hold_seconds > 1.0:
				if self._can_count():
					st.reps += 1
			st.hold_start = None
			st.hold_seconds = 0.0
			st.state = "rest"
		return "Front Bridge", st, f"Hold: {st.hold_seconds:.1f}s"

	def _detect_alternate_leg_lowers(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["alternate_leg_lowers"]
		knee_l = angle_3pt(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
		side = None
		if knee_l > 160 and knee_r < 120:
			side = "L"
		elif knee_r > 160 and knee_l < 120:
			side = "R"
		if side and side != st.last_side:
			if self._can_count():
				st.reps += 1
			st.last_side = side
		return "Bent Knee Alternate Leg Lowers", st, f"Side: {side or ''}"

	def _detect_thoracic_rotation(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["thoracic_rotation"]
		shoulder_z_diff = lm["shoulder_l"][2] - lm["shoulder_r"][2]
		now = time.time()
		if shoulder_z_diff > 0.12 and st.state != "left":
			st.state = "left"
			if self._can_count():
				st.reps += 1
			st.last_transition_time = now
		elif shoulder_z_diff < -0.12 and st.state != "right":
			st.state = "right"
			if self._can_count():
				st.reps += 1
			st.last_transition_time = now
		return "Thoracic Rotation", st, f"Δz: {shoulder_z_diff:.2f}"


def extract_landmarks(landmarks) -> Dict[str, np.ndarray]:
	def pt(index: int) -> np.ndarray:
		return np.array([landmarks[index].x, landmarks[index].y, landmarks[index].z], dtype=np.float32)

	return {
		"nose": pt(0),
		"shoulder_l": pt(11),
		"shoulder_r": pt(12),
		"elbow_l": pt(13),
		"elbow_r": pt(14),
		"wrist_l": pt(15),
		"wrist_r": pt(16),
		"hip_l": pt(23),
		"hip_r": pt(24),
		"knee_l": pt(25),
		"knee_r": pt(26),
		"ankle_l": pt(27),
		"ankle_r": pt(28),
	}


def draw_landmarks_simple(image, landmarks) -> None:
	h, w = image.shape[:2]
	for lm in landmarks:
		x = int(lm.x * w)
		y = int(lm.y * h)
		cv2.circle(image, (x, y), 3, (0, 255, 0), -1)


def draw_text(img, text: str, org: Tuple[int, int], color=(0, 255, 0)) -> None:
	cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


EXERCISE_LABELS = {
	"squat": "Squats",
	"lateral_lunge": "Dynamic Hip Mobility Lateral Lunge",
	"chest_press": "Chest Press",
	"vertical_traction": "Vertical Traction",
	"hip_raise": "Bent Knee Hip Raise",
	"airplane": "Airplane Exercise",
	"front_bridge": "Front Bridge",
	"alternate_leg_lowers": "Bent Knee Alternate Leg Lowers",
	"thoracic_rotation": "Thoracic Rotation",
}


def draw_status_window(
	detector: "ExerciseDetector",
	last_feedback_by_exercise: Dict[str, str],
	last_feedback_time: Dict[str, float],
	now: float,
	exercises: Tuple[str, ...] = ("squat", "lateral_lunge"),
	window_name: str = "CareSystem AI - Status",
) -> None:
	line_height = 22
	margin_x = 20
	margin_y = 24
	height = margin_y * 2 + (len(exercises) * 2 + 1) * line_height
	width = 560
	canvas = np.zeros((height, width, 3), dtype=np.uint8)
	text_color = (255, 255, 255)
	cv2.putText(
		canvas,
		"Reps & Feedback",
		(margin_x, margin_y),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		text_color,
		2,
		cv2.LINE_AA,
	)
	y = margin_y + line_height
	for key in exercises:
		label = EXERCISE_LABELS.get(key, key.replace("_", " ").title())
		reps = detector.states[key].reps
		cv2.putText(
			canvas,
			f"{label}: {reps}",
			(margin_x, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			text_color,
			1,
			cv2.LINE_AA,
		)
		y += line_height
		feedback = ""
		if (now - last_feedback_time.get(key, 0.0)) <= 2.0:
			feedback = last_feedback_by_exercise.get(key, "")
		if feedback:
			cv2.putText(
				canvas,
				feedback,
				(margin_x + 18, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				text_color,
				1,
				cv2.LINE_AA,
			)
		y += line_height
	cv2.imshow(window_name, canvas)


def _landmark_visible(lm, min_visibility: float) -> bool:
	visibility = getattr(lm, "visibility", None)
	if visibility is not None and visibility < min_visibility:
		return False
	return 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0


def full_body_visible(landmarks, min_visibility: float = 0.5, min_height_ratio: float = 0.45) -> bool:
	required = [11, 12, 23, 24, 27, 28]
	for idx in required:
		if not _landmark_visible(landmarks[idx], min_visibility):
			return False
	shoulder_y = (landmarks[11].y + landmarks[12].y) / 2.0
	ankle_y = (landmarks[27].y + landmarks[28].y) / 2.0
	return abs(ankle_y - shoulder_y) >= min_height_ratio


def main() -> None:
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam.")

	detector = ExerciseDetector()
	face_id = FaceIDManager()
	speaker = ElevenLabsSpeaker()
	face_id.load_db()
	last_id_time = 0.0
	last_sync_time = 0.0
	last_rep_time: Optional[float] = None
	last_total_reps = 0
	last_status_log = 0.0
	last_candidate: Optional[str] = None
	candidate_hits = 0
	last_activity_time: Optional[float] = None
	last_active_user: Optional[str] = None
	last_feedback_by_exercise = {key: "" for key in detector.states}
	last_feedback_time = {key: 0.0 for key in detector.states}
	feedback_cooldown = 3.0
	detector.reset_exercises({"squat", "lateral_lunge"})
	if not USE_TASKS:
		with mp_pose.Pose(
			model_complexity=1,
			enable_segmentation=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		) as pose:
			while True:
				ret, frame = cap.read()
				if not ret:
					break

				now = time.time()
				if last_activity_time is not None and (now - last_activity_time) > 10:
					face_id.active_user = None
				if face_id.active_user is None and (now - last_id_time > 1.5):
					candidate = face_id.identify(frame)
					last_id_time = now
					if candidate:
						if candidate == last_candidate:
							candidate_hits += 1
						else:
							last_candidate = candidate
							candidate_hits = 1
						if candidate_hits >= 2:
							face_id.active_user = candidate
							last_candidate = None
							candidate_hits = 0
				if face_id.active_user != last_active_user:
					if face_id.active_user is not None:
						detector.reset_exercises({"squat", "lateral_lunge"})
						last_total_reps = 0
					last_active_user = face_id.active_user
				detector.allow_count = face_id.active_user is not None
				if face_id.active_user is None:
					for key in last_feedback_by_exercise:
						last_feedback_by_exercise[key] = ""
						last_feedback_time[key] = 0.0
				if now - last_status_log > 3:
					print(f"[FaceID] {face_id.get_status()}")
					last_status_log = now
				detector.allow_count = face_id.active_user is not None

				image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				image.flags.writeable = False
				results = pose.process(image)
				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				h, w = image.shape[:2]
				status_text = "No pose"
				if results.pose_landmarks:
					if mp_drawing and POSE_CONNECTIONS:
						mp_drawing.draw_landmarks(
							image, results.pose_landmarks, POSE_CONNECTIONS
						)
					if full_body_visible(results.pose_landmarks.landmark):
						last_activity_time = now
						lm = extract_landmarks(results.pose_landmarks.landmark)
						ex_name, ex_state, status_text = detector.process(lm)
						if detector.exercise in ("squat", "lateral_lunge") and face_id.active_user is not None:
							parts = [p for p in status_text.split(" | ") if not p.startswith("Conf:")]
							feedback = parts[1] if len(parts) > 1 else ""
							if feedback:
								prev_feedback = last_feedback_by_exercise[detector.exercise]
								cooldown_ok = (now - last_feedback_time[detector.exercise]) >= feedback_cooldown
								if feedback != prev_feedback or cooldown_ok:
									last_feedback_by_exercise[detector.exercise] = feedback
									last_feedback_time[detector.exercise] = now
									speaker.say_feedback(detector.exercise, feedback)
						full_body_ok = True
					else:
						full_body_ok = False
						ex_name = "Detecting..."
						ex_state = detector.states["squat"]
						status_text = "Step back to show full body"
					total_reps = detector.states["squat"].reps + detector.states["lateral_lunge"].reps
					if total_reps != last_total_reps:
						last_total_reps = total_reps
						last_rep_time = now
						last_activity_time = now
					if face_id.active_user and full_body_ok and (now - last_sync_time > 5):
						face_id.log_workout(ex_name, ex_state.reps)
						last_sync_time = now
					draw_text(image, f"User: {face_id.active_user or 'Scanning...'}", (20, 30), (255, 255, 0))
					draw_text(image, f"Exercise: {ex_name}", (20, 60))
					draw_text(image, f"Reps: {ex_state.reps}", (20, 90))
					draw_text(image, f"State: {ex_state.state}", (20, 120))
					draw_text(image, status_text, (20, 150))
				else:
					draw_text(image, f"User: {face_id.active_user or 'Scanning...'}", (20, 30), (255, 255, 0))
					draw_text(image, "No pose detected", (20, 60), (0, 0, 255))

				draw_text(image, "Keys: A Auto | 1 Squat | 2 Lunge | Q Quit", (20, h - 20))
				draw_status_window(detector, last_feedback_by_exercise, last_feedback_time, now)

				cv2.imshow("CareSystem AI - CV", image)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				if key == ord("a"):
					detector.set_auto(True)
				if key == ord("1"):
					detector.set_exercise("squat")
				if key == ord("2"):
					detector.set_exercise("lateral_lunge")
				
	else:
		from mediapipe.tasks import python
		from mediapipe.tasks.python import vision

		def get_pose_model_path() -> str:
			model_dir = Path(__file__).resolve().parent / "models"
			model_dir.mkdir(parents=True, exist_ok=True)
			model_path = model_dir / "pose_landmarker.task"
			if model_path.exists():
				return str(model_path)
			url = (
				"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
				"pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
			)
			try:
				urlretrieve(url, model_path)
			except Exception as exc:
				raise FileNotFoundError(
					"Unable to download pose_landmarker.task. "
					"Check your internet connection or place the file at models/pose_landmarker.task"
				) from exc
			return str(model_path)

		base_options = python.BaseOptions(model_asset_path=get_pose_model_path())
		options = vision.PoseLandmarkerOptions(
			base_options=base_options,
			running_mode=vision.RunningMode.VIDEO,
			num_poses=1,
			min_pose_detection_confidence=0.5,
			min_pose_presence_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		with vision.PoseLandmarker.create_from_options(options) as landmarker:
			while True:
				ret, frame = cap.read()
				if not ret:
					break

				now = time.time()
				if last_activity_time is not None and (now - last_activity_time) > 10:
					face_id.active_user = None
				if face_id.active_user is None and (now - last_id_time > 1.5):
					candidate = face_id.identify(frame)
					last_id_time = now
					if candidate:
						if candidate == last_candidate:
							candidate_hits += 1
						else:
							last_candidate = candidate
							candidate_hits = 1
						if candidate_hits >= 2:
							face_id.active_user = candidate
							last_candidate = None
							candidate_hits = 0
				if now - last_status_log > 3:
					print(f"[FaceID] {face_id.get_status()}")
					last_status_log = now
				detector.allow_count = face_id.active_user is not None
				if face_id.active_user is None:
					for key in last_feedback_by_exercise:
						last_feedback_by_exercise[key] = ""
						last_feedback_time[key] = 0.0

				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
				ts = int(time.time() * 1000)
				result = landmarker.detect_for_video(mp_image, ts)
				image = frame.copy()

				h, w = image.shape[:2]
				status_text = "No pose"
				if result.pose_landmarks:
					landmarks = result.pose_landmarks[0]
					draw_landmarks_simple(image, landmarks)
					last_activity_time = now
					lm = extract_landmarks(landmarks)
					ex_name, ex_state, status_text = detector.process(lm)
					full_body_ok = full_body_visible(landmarks)
					if detector.exercise in ("squat", "lateral_lunge") and face_id.active_user is not None:
						parts = [p for p in status_text.split(" | ") if not p.startswith("Conf:")]
						feedback = parts[1] if len(parts) > 1 else ""
						if feedback:
							prev_feedback = last_feedback_by_exercise[detector.exercise]
							cooldown_ok = (now - last_feedback_time[detector.exercise]) >= feedback_cooldown
							if feedback != prev_feedback or cooldown_ok:
								last_feedback_by_exercise[detector.exercise] = feedback
								last_feedback_time[detector.exercise] = now
								if full_body_ok:
									speaker.say_feedback(detector.exercise, feedback)
					total_reps = detector.states["squat"].reps + detector.states["lateral_lunge"].reps
					if total_reps != last_total_reps:
						last_total_reps = total_reps
						last_rep_time = now
						last_activity_time = now
					if face_id.active_user and ex_name != "Detecting..." and (now - last_sync_time > 5):
						face_id.log_workout(ex_name, ex_state.reps)
						last_sync_time = now
					draw_text(image, f"User: {face_id.active_user or 'Scanning...'}", (20, 30), (255, 255, 0))
					draw_text(image, f"Exercise: {ex_name}", (20, 60))
					draw_text(image, f"Reps: {ex_state.reps}", (20, 90))
					draw_text(image, f"State: {ex_state.state}", (20, 120))
					draw_text(image, status_text, (20, 150))
				else:
					draw_text(image, f"User: {face_id.active_user or 'Scanning...'}", (20, 30), (255, 255, 0))
					draw_text(image, "No pose detected", (20, 60), (0, 0, 255))

				draw_text(image, "Keys: A Auto | 1 Squat | 2 Lunge | Q Quit", (20, h - 20))
				draw_status_window(detector, last_feedback_by_exercise, last_feedback_time, now)

				cv2.imshow("CareSystem AI - CV", image)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				if key == ord("a"):
					detector.set_auto(True)
				if key == ord("1"):
					detector.set_exercise("squat")
				if key == ord("2"):
					detector.set_exercise("lateral_lunge")
			

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
