import { useEffect, useState } from "react";

export default function PresageTest() {
  const [status, setStatus] = useState("connecting...");

  useEffect(() => {
    const wsUrl = import.meta.env.VITE_PRESAGE_WS || "ws://localhost:8080/ws";
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setStatus("connected");
      console.log("Presage WS open");
    };

    ws.onmessage = (e) => {
      console.log("PRESAGE MSG:", e.data);
    };

    ws.onerror = () => {
      setStatus("error");
      console.log("Presage WS error");
    };

    ws.onclose = () => {
      setStatus("closed");
      console.log("Presage WS closed");
    };

    return () => ws.close();
  }, []);

  return (
    <div style={{ padding: 16 }}>
      <h3>Presage Test</h3>
      <p>Status: {status}</p>
      <p>Open DevTools → Console to see messages.</p>
    </div>
  );
}
