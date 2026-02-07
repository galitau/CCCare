// src/hooks/usePresage.js
ws.onmessage = (event) => {
  console.log("RAW PRESAGE MESSAGE:", event.data);
  // existing code below
};

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * usePresage({ isActive, exercise, sessionId })
 *
 * Connects to Presage MetricsGateway WebSocket and provides:
 * - heartRate (number | null)
 * - breathingRate (number | null)
 * - breathingConsistency ("steady" | "variable" | null)
 * - signalConfidence ("low" | "medium" | "high")
 * - startWorkout(), stopWorkout() -> also saves to Mongo via your backend
 * - workoutTimeMs
 *
 * IMPORTANT: This hook expects Presage OnPrem gateway to be running:
 *   VITE_PRESAGE_WS=ws://localhost:8080/ws
 */
export function usePresage({ isActive, exercise, sessionId }) {
  // Presage WS
  const PRESAGE_WS = import.meta.env.VITE_PRESAGE_WS || "ws://localhost:8080/ws";
  // Your own backend that writes to MongoDB
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:3000";

  const wsRef = useRef(null);
  const lastVitalsMsRef = useRef(0);

  // Timer
  const [workoutRunning, setWorkoutRunning] = useState(false);
  const [startMs, setStartMs] = useState(null);
  const [workoutTimeMs, setWorkoutTimeMs] = useState(0);

  // Vitals
  const [heartRate, setHeartRate] = useState(null);
  const [breathingRate, setBreathingRate] = useState(null);
  const [signalConfidence, setSignalConfidence] = useState("low");
  const [breathingConsistency, setBreathingConsistency] = useState(null);

  // Store samples for MongoDB
  const samplesRef = useRef([]); // {tMs, hr, br}

  // Breathing consistency window
  const brWindowRef = useRef([]);

  const confidenceFromAge = (ageMs) => {
    if (ageMs < 1500) return "high";
    if (ageMs < 4000) return "medium";
    return "low";
  };

  const safeRound = (v) => (typeof v === "number" && Number.isFinite(v) ? Math.round(v) : null);

  const pickRate = (values, regex) => {
    if (!Array.isArray(values)) return null;
    const item = values.find((v) => regex.test(String(v.label || "")));
    return safeRound(item?.value);
  };

  const updateConsistency = (br) => {
    const w = brWindowRef.current;
    w.push(br);
    if (w.length > 8) w.shift();

    if (w.length < 5) {
      setBreathingConsistency(null);
      return;
    }

    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    const variance = w.reduce((a, b) => a + (b - mean) ** 2, 0) / w.length;
    const std = Math.sqrt(variance);
    const cv = mean > 0 ? std / mean : 1;

    setBreathingConsistency(cv < 0.12 ? "steady" : "variable");
  };

  const sendWs = (obj) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(obj));
  };

  const connectWs = useCallback(() => {
    // Only connect when active
    if (!isActive) return;

    // Already open
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(PRESAGE_WS);
    wsRef.current = ws;

    ws.onopen = () => {
      // Start recording (supported by Presage gateway) 
      sendWs({ type: "set_recording", value: true });
    };

    ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }

      // Gateway streams vitals in `rate_update` 
      if (msg.type === "rate_update" && Array.isArray(msg.values)) {
        const hr = pickRate(msg.values, /heart|pulse/i);
        const br = pickRate(msg.values, /breath|resp/i);

        if (hr !== null) setHeartRate(hr);
        if (br !== null) {
          setBreathingRate(br);
          updateConsistency(br);
        }

        lastVitalsMsRef.current = Date.now();
        setSignalConfidence("high");

        // If workout running, record sample (even if only one of hr/br exists)
        if (workoutRunning) {
          samplesRef.current.push({
            tMs: Date.now(),
            hr: hr,
            br: br,
          });
        }
      }
    };

    ws.onerror = () => setSignalConfidence("low");
    ws.onclose = () => setSignalConfidence("low");
  }, [PRESAGE_WS, isActive, workoutRunning]);

  // Maintain confidence decay if vitals stop updating
  useEffect(() => {
    if (!isActive) return;
    const id = setInterval(() => {
      const age = Date.now() - (lastVitalsMsRef.current || 0);
      setSignalConfidence(confidenceFromAge(age));
    }, 500);
    return () => clearInterval(id);
  }, [isActive]);

  // Timer loop
  useEffect(() => {
    if (!workoutRunning || startMs === null) return;
    const id = setInterval(() => {
      setWorkoutTimeMs(Date.now() - startMs);
    }, 250);
    return () => clearInterval(id);
  }, [workoutRunning, startMs]);

  // Connect/disconnect WS based on isActive
  useEffect(() => {
    if (!isActive) {
      // Stop recording + cleanup
      try {
        sendWs({ type: "set_recording", value: false });
      } catch {}
      wsRef.current?.close();
      wsRef.current = null;

      // Reset vitals UI
      setHeartRate(null);
      setBreathingRate(null);
      setSignalConfidence("low");
      setBreathingConsistency(null);

      // Stop workout
      setWorkoutRunning(false);
      setStartMs(null);
      setWorkoutTimeMs(0);

      samplesRef.current = [];
      brWindowRef.current = [];
      return;
    }

    connectWs();

    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [isActive, connectWs]);

  const postJson = async (path, body) => {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API error ${res.status}`);
    return res.json().catch(() => ({}));
  };

  const startWorkout = useCallback(async () => {
    samplesRef.current = [];
    brWindowRef.current = [];

    const now = Date.now();
    setWorkoutRunning(true);
    setStartMs(now);
    setWorkoutTimeMs(0);

    // Optional: create session doc immediately
    await postJson("/api/workouts/start", {
      sessionId,
      exercise: exercise || null,
      startTimeMs: now,
    });
  }, [sessionId, exercise, API_BASE]);

  const stopWorkout = useCallback(async () => {
    const end = Date.now();
    const start = startMs ?? end;
    const durationMs = end - start;

    setWorkoutRunning(false);

    // Save final workout to MongoDB
    await postJson("/api/workouts/finish", {
      sessionId,
      exercise: exercise || null,
      startTimeMs: start,
      endTimeMs: end,
      durationMs,
      latest: {
        hr: heartRate,
        br: breathingRate,
        confidence: signalConfidence,
        breathingConsistency,
      },
      samples: samplesRef.current,
      source: "presage_ws",
    });

    setStartMs(null);
    setWorkoutTimeMs(0);
  }, [sessionId, exercise, startMs, heartRate, breathingRate, signalConfidence, breathingConsistency]);

  return {
    heartRate,
    breathingRate,
    breathingConsistency,
    signalConfidence,
    workoutTimeMs,
    workoutRunning,
    startWorkout,
    stopWorkout,
  };
}
