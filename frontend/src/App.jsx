import { useState } from "react";
import { HeartPulse, Play, Square, User } from "lucide-react";

const patients = [
  {
    id: "p1",
    name: "Jane Doe",
    heartRate: 92,
    oxygen: 97,
    breathingRate: 16,
    trainingScore: 72,
    history: [
      { date: "2026-02-01", avgHR: 88, reps: 24, score: 68 },
      { date: "2026-02-03", avgHR: 91, reps: 30, score: 74 },
    ],
  },
  {
    id: "p2",
    name: "John Smith",
    heartRate: 110,
    oxygen: 94,
    breathingRate: 20,
    trainingScore: 55,
    history: [{ date: "2026-02-03", avgHR: 105, reps: 18, score: 52 }],
  },
];

function Stat({ label, value, icon, alert }) {
  return (
    <div
      className="border rounded-xl p-4"
      style={{ borderColor: alert ? "#ef4444" : "#e5e7eb" }}
    >
      <div className="flex items-center gap-2 text-sm text-gray-500">
        {icon} {label}
      </div>
      <div className="text-lg font-bold">{value}</div>
      {alert && (
        <div className="text-xs text-red-600 mt-1">
          Above threshold
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);

  const filtered = patients.filter((p) =>
    p.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6 font-sans">
      <header className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">
          CCCARE Coach Dashboard
        </h1>

        {!sessionActive ? (
          <button
            onClick={() => setSessionActive(true)}
            className="px-4 py-2 rounded-lg border flex items-center gap-2"
          >
            <Play size={16} />
            Start Program
          </button>
        ) : (
          <button
            onClick={() => setSessionActive(false)}
            className="px-4 py-2 rounded-lg border border-red-500 text-red-600 flex items-center gap-2"
          >
            <Square size={16} />
            Stop Program
          </button>
        )}
      </header>

      {/* Search + Patient List */}
      <div className="border rounded-xl p-4 space-y-4">
        <input
          placeholder="Search patient from attendance list..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full border rounded-lg p-2"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {filtered.map((patient) => (
            <button
              key={patient.id}
              onClick={() => setSelectedPatient(patient)}
              className="border rounded-lg p-2 flex items-center gap-2 text-left"
            >
              <User size={16} />
              {patient.name}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Patient */}
      {selectedPatient && (
        <div className="border rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">
            {selectedPatient.name}
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat
              label="Heart Rate"
              value={`${selectedPatient.heartRate} bpm`}
              icon={<HeartPulse size={16} />}
              alert={selectedPatient.heartRate > 100}
            />
            <Stat
              label="Oxygen"
              value={`${selectedPatient.oxygen}%`}
            />
            <Stat
              label="Breathing"
              value={`${selectedPatient.breathingRate} /min`}
            />
            <Stat
              label="Training Score"
              value={selectedPatient.trainingScore}
            />
          </div>

          <div>
            <h3 className="font-semibold mb-2">
              Session History
            </h3>

            <div className="space-y-2">
              {selectedPatient.history.map((h, i) => (
                <div
                  key={i}
                  className="flex flex-wrap justify-between border p-2 rounded text-sm"
                >
                  <span>{h.date}</span>
                  <span>Avg HR: {h.avgHR}</span>
                  <span>Reps: {h.reps}</span>
                  <span className="border px-2 rounded-full">
                    Score: {h.score}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
