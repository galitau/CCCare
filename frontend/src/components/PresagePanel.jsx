import { useEffect, useMemo, useState } from "react";

function formatMmSs(ms) {
  const totalSec = Math.floor(ms / 1000);
  const mm = Math.floor(totalSec / 60);
  const ss = totalSec % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export default function PresagePanel() {
  const [heartRate, setHeartRate] = useState(null);
  const [breathingRate, setBreathingRate] = useState(null);
  const [confidence, setConfidence] = useState("unknown");
  const [startTime, setStartTime] = useState(null);

  const sessionId = useMemo(() => crypto.randomUUID(), []);

  useEffect(() => {
    const wsUrl =
      import.meta.env.VITE_PRESAGE_WS || "ws://localhost:8080/ws";

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("Presage WS connected");
      setStartTime(Date.now());
    };

    ws.onmessage = (event) => {
      console.log("RAW PRESAGE MESSAGE:", event.data);

      // Try parsing JSON safely
      try {
        const data = JSON.parse(event.data);

        // These keys depend on Presage output — we log first, map later
        if (data.heartRate) setHeartRate(data.heartRate);
        if (data.breathingRate) setBreathingRate(data.breathingRate);
        if (data.confidence) setConfidence(data.confidence);
      } catch (err) {
        // Some Presage messages may not be JSON
      }
    };

    ws.onerror = (err) => {
      console.error("Presage WS error", err);
    };

    ws.onclose = () => {
      console.log("Presage WS closed");
    };

    return () => ws.close();
  }, [sessionId]);

  const workoutTimeMs = startTime ? Date.now() - startTime : 0;

  return (
    <div style={{ padding: 16, maxWidth: 420 }}>
      <h3>Presage Debug Panel</h3>

      <div style={{ padding: 12, border: "1px solid #ccc", borderRadius: 8 }}>
        <div>Heart Rate: <b>{heartRate ?? "--"}</b> bpm</div>
        <div>Breathing Rate: <b>{breathingRate ?? "--"}</b> /min</div>
        <div>Signal Confidence: <b>{confidence}</b></div>
        <div>Workout Time: <b>{formatMmSs(workoutTimeMs)}</b></div>
      </div>

      <p style={{ marginTop: 12, fontSize: 12, color: "#666" }}>
        Open DevTools → Console to see raw Presage messages.
      </p>
    </div>
  );
}
