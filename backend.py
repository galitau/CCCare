import time
from pathlib import Path
from urllib.request import urlretrieve
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

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


class ExerciseDetector:
	def __init__(self) -> None:
		self.exercise = "squat"
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

	def set_exercise(self, name: str) -> None:
		if name in self.states:
			self.exercise = name

	def process(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		if self.exercise == "squat":
			return self._detect_squat(lm)
		if self.exercise == "lateral_lunge":
			return self._detect_lateral_lunge(lm)
		if self.exercise == "chest_press":
			return self._detect_chest_press(lm)
		if self.exercise == "vertical_traction":
			return self._detect_vertical_traction(lm)
		if self.exercise == "hip_raise":
			return self._detect_hip_raise(lm)
		if self.exercise == "airplane":
			return self._detect_airplane(lm)
		if self.exercise == "front_bridge":
			return self._detect_front_bridge(lm)
		if self.exercise == "alternate_leg_lowers":
			return self._detect_alternate_leg_lowers(lm)
		if self.exercise == "thoracic_rotation":
			return self._detect_thoracic_rotation(lm)
		return self.exercise, self.states[self.exercise], ""

	def _transition(self, state: ExerciseState, new_state: str, now: float) -> None:
		if state.state != new_state:
			state.state = new_state
			state.last_transition_time = now

	def _detect_squat(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["squat"]
		knee_l = angle_3pt_2d(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt_2d(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
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
			st.reps += 1
			self._transition(st, "up", now)

		return "Squats", st, f"Knee: {st.ema_knee:.0f}°"

	def _detect_lateral_lunge(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["lateral_lunge"]
		knee_l = angle_3pt(lm["hip_l"], lm["knee_l"], lm["ankle_l"])
		knee_r = angle_3pt(lm["hip_r"], lm["knee_r"], lm["ankle_r"])
		hip_center = midpoint(lm["hip_l"], lm["hip_r"])
		ankle_mid = midpoint(lm["ankle_l"], lm["ankle_r"])
		lateral_shift = hip_center[0] - ankle_mid[0]
		now = time.time()
		down_left = knee_l < 105 and knee_r > 150 and lateral_shift < -0.03
		down_right = knee_r < 105 and knee_l > 150 and lateral_shift > 0.03
		if down_left or down_right:
			self._transition(st, "down", now)
		if knee_l > 160 and knee_r > 160 and st.state == "down":
			st.reps += 1
			self._transition(st, "up", now)
		side = "L" if down_left else "R" if down_right else ""
		return "Dynamic Hip Mobility Lateral Lunge", st, f"Side: {side}"

	def _detect_chest_press(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["chest_press"]
		elbow_l = angle_3pt(lm["shoulder_l"], lm["elbow_l"], lm["wrist_l"])
		elbow_r = angle_3pt(lm["shoulder_r"], lm["elbow_r"], lm["wrist_r"])
		avg_elbow = (elbow_l + elbow_r) / 2.0
		now = time.time()
		if avg_elbow < 95:
			self._transition(st, "down", now)
		if avg_elbow > 160 and st.state == "down":
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
			st.reps += 1
			st.last_side = side
		return "Bent Knee Alternate Leg Lowers", st, f"Side: {side or ''}"

	def _detect_thoracic_rotation(self, lm: Dict[str, np.ndarray]) -> Tuple[str, ExerciseState, str]:
		st = self.states["thoracic_rotation"]
		shoulder_z_diff = lm["shoulder_l"][2] - lm["shoulder_r"][2]
		now = time.time()
		if shoulder_z_diff > 0.12 and st.state != "left":
			st.state = "left"
			st.reps += 1
			st.last_transition_time = now
		elif shoulder_z_diff < -0.12 and st.state != "right":
			st.state = "right"
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


def main() -> None:
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam.")

	detector = ExerciseDetector()
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
					lm = extract_landmarks(results.pose_landmarks.landmark)
					ex_name, ex_state, status_text = detector.process(lm)
					draw_text(image, f"Exercise: {ex_name}", (20, 30))
					draw_text(image, f"Reps: {ex_state.reps}", (20, 60))
					draw_text(image, f"State: {ex_state.state}", (20, 90))
					draw_text(image, status_text, (20, 120))
				else:
					draw_text(image, "No pose detected", (20, 30), (0, 0, 255))

				draw_text(image, "Keys: 1 Squat | 2 Lunge | 3 Chest Press | 4 Traction | 5 Hip Raise", (20, h - 50))
				draw_text(image, "6 Airplane | 7 Front Bridge | 8 Alt Leg Lowers | 9 Thoracic Rotation | Q Quit", (20, h - 20))

				cv2.imshow("CareSystem AI - CV", image)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				if key == ord("1"):
					detector.set_exercise("squat")
				if key == ord("2"):
					detector.set_exercise("lateral_lunge")
				if key == ord("3"):
					detector.set_exercise("chest_press")
				if key == ord("4"):
					detector.set_exercise("vertical_traction")
				if key == ord("5"):
					detector.set_exercise("hip_raise")
				if key == ord("6"):
					detector.set_exercise("airplane")
				if key == ord("7"):
					detector.set_exercise("front_bridge")
				if key == ord("8"):
					detector.set_exercise("alternate_leg_lowers")
				if key == ord("9"):
					detector.set_exercise("thoracic_rotation")
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
					lm = extract_landmarks(landmarks)
					ex_name, ex_state, status_text = detector.process(lm)
					draw_text(image, f"Exercise: {ex_name}", (20, 30))
					draw_text(image, f"Reps: {ex_state.reps}", (20, 60))
					draw_text(image, f"State: {ex_state.state}", (20, 90))
					draw_text(image, status_text, (20, 120))
				else:
					draw_text(image, "No pose detected", (20, 30), (0, 0, 255))

				draw_text(image, "Keys: 1 Squat | 2 Lunge | 3 Chest Press | 4 Traction | 5 Hip Raise", (20, h - 50))
				draw_text(image, "6 Airplane | 7 Front Bridge | 8 Alt Leg Lowers | 9 Thoracic Rotation | Q Quit", (20, h - 20))

				cv2.imshow("CareSystem AI - CV", image)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				if key == ord("1"):
					detector.set_exercise("squat")
				if key == ord("2"):
					detector.set_exercise("lateral_lunge")
				if key == ord("3"):
					detector.set_exercise("chest_press")
				if key == ord("4"):
					detector.set_exercise("vertical_traction")
				if key == ord("5"):
					detector.set_exercise("hip_raise")
				if key == ord("6"):
					detector.set_exercise("airplane")
				if key == ord("7"):
					detector.set_exercise("front_bridge")
				if key == ord("8"):
					detector.set_exercise("alternate_leg_lowers")
				if key == ord("9"):
					detector.set_exercise("thoracic_rotation")

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
