import { useState, useEffect, useRef } from 'react';
import './App.css';

export default function App() {
  // State variables
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [heartRate, setHeartRate] = useState('--');
  const [breathing, setBreathing] = useState('--');
  const [oxygen, setOxygen] = useState('--');
  const [showHeartRateAlert, setShowHeartRateAlert] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [currentAngle, setCurrentAngle] = useState('0°');
  const [postureStatus, setPostureStatus] = useState({ good: true, text: '✅ Good Form' });
  const [exerciseQuality, setExerciseQuality] = useState('92%');
  const [patients, setPatients] = useState([
    {
      id: 1,
      name: "John Smith",
      age: 68,
      condition: "Stroke Recovery",
      medicalRecords: {
        diagnosis: "Ischemic Stroke (2023)",
        medications: ["Aspirin 81mg", "Atorvastatin 20mg"],
        allergies: ["Penicillin"],
        bloodType: "O+",
        emergencyContact: "Jane Smith - 555-0123"
      },
      history: [
        {
          date: new Date(2026, 1, 5),
          duration: "30 minutes",
          avgHeartRate: 78,
          maxHeartRate: 95,
          avgOxygen: 97,
          exerciseCompleted: "Arm raises, Leg lifts",
          trainingEffect: 3.2,
          notes: "Good session, maintained proper form"
        }
      ]
    },
    {
      id: 2,
      name: "Mary Johnson",
      age: 72,
      condition: "Dementia",
      medicalRecords: {
        diagnosis: "Alzheimer's Disease (2021)",
        medications: ["Donepezil 10mg", "Memantine 10mg"],
        allergies: ["None"],
        bloodType: "A+",
        emergencyContact: "Robert Johnson - 555-0456"
      },
      history: []
    },
    {
      id: 3,
      name: "Robert Davis",
      age: 65,
      condition: "Cardiac Rehabilitation",
      medicalRecords: {
        diagnosis: "Myocardial Infarction (2024)",
        medications: ["Metoprolol 50mg", "Lisinopril 10mg", "Clopidogrel 75mg"],
        allergies: ["Sulfa drugs"],
        bloodType: "B+",
        emergencyContact: "Linda Davis - 555-0789"
      },
      history: []
    },
    {
      id: 4,
      name: "Patricia Wilson",
      age: 70,
      condition: "Parkinson's Disease",
      medicalRecords: {
        diagnosis: "Parkinson's Disease (2020)",
        medications: ["Carbidopa-Levodopa 25-100mg", "Pramipexole 0.5mg"],
        allergies: ["Latex"],
        bloodType: "AB+",
        emergencyContact: "Michael Wilson - 555-0321"
      },
      history: []
    },
    {
      id: 5,
      name: "James Brown",
      age: 69,
      condition: "Post-Surgery Recovery",
      medicalRecords: {
        diagnosis: "Hip Replacement Surgery (2025)",
        medications: ["Acetaminophen 500mg", "Enoxaparin 40mg"],
        allergies: ["Iodine"],
        bloodType: "O-",
        emergencyContact: "Sarah Brown - 555-0654"
      },
      history: []
    }
  ]);

  const vitalsIntervalRef = useRef(null);
  const exerciseIntervalRef = useRef(null);
  const webcamStreamRef = useRef(null);
  const videoRef = useRef(null);

  // Clean up intervals on unmount
  useEffect(() => {
    return () => {
      if (vitalsIntervalRef.current) clearInterval(vitalsIntervalRef.current);
      if (exerciseIntervalRef.current) clearInterval(exerciseIntervalRef.current);
      if (webcamStreamRef.current) {
        webcamStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startVitalsMonitoring = () => {
    vitalsIntervalRef.current = setInterval(() => {
      if (!selectedPatient || !sessionActive) return;

      const hr = Math.floor(60 + Math.random() * 40);
      const br = Math.floor(12 + Math.random() * 8);
      const ox = Math.floor(95 + Math.random() * 5);

      setHeartRate(hr);
      setBreathing(br);
      setOxygen(ox);

      const maxHeartRate = 220 - selectedPatient.age;
      const targetHeartRate = maxHeartRate * 0.7;

      if (hr > targetHeartRate) {
        setShowHeartRateAlert(true);
      } else {
        setShowHeartRateAlert(false);
      }
    }, 2000);
  };

  const stopVitalsMonitoring = () => {
    if (vitalsIntervalRef.current) {
      clearInterval(vitalsIntervalRef.current);
      vitalsIntervalRef.current = null;
    }
  };

  const startExerciseTracking = () => {
    let repCount = 0;
    
    exerciseIntervalRef.current = setInterval(() => {
      if (!sessionActive) return;

      if (Math.random() > 0.7) {
        repCount++;
        setRepCount(repCount);
      }

      const angle = Math.floor(Math.random() * 180);
      setCurrentAngle(angle + '°');

      if (angle >= 60 && angle <= 120) {
        setPostureStatus({ good: true, text: '✅ Good Form' });
        setExerciseQuality('92%');
      } else {
        setPostureStatus({ good: false, text: '⚠️ Adjust Posture' });
        setExerciseQuality('65%');
      }
    }, 2000);
  };

  const stopExerciseTracking = () => {
    if (exerciseIntervalRef.current) {
      clearInterval(exerciseIntervalRef.current);
      exerciseIntervalRef.current = null;
    }
    setRepCount(0);
    setCurrentAngle('0°');
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 }
      });
      webcamStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.log('Webcam not available:', error);
    }
  };

  const stopWebcam = () => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach(track => track.stop());
      webcamStreamRef.current = null;
    }
  };

  const startSession = () => {
    setSessionActive(true);
    if (selectedPatient) {
      startVitalsMonitoring();
      startExerciseTracking();
      startWebcam();
    }
  };

  const stopSession = () => {
    setSessionActive(false);
    stopVitalsMonitoring();
    stopExerciseTracking();
    stopWebcam();

    setHeartRate('--');
    setBreathing('--');
    setOxygen('--');
    setShowHeartRateAlert(false);

    if (selectedPatient) {
      const newSession = {
        date: new Date(),
        duration: "30 minutes",
        avgHeartRate: 78,
        maxHeartRate: 95,
        avgOxygen: 97,
        exerciseCompleted: "Arm raises, Leg lifts",
        trainingEffect: 3.2,
        notes: "Good session, maintained proper form"
      };

      setPatients(patients.map(p => 
        p.id === selectedPatient.id 
          ? { ...p, history: [newSession, ...p.history] }
          : p
      ));

      setSelectedPatient(prev => ({
        ...prev,
        history: [newSession, ...prev.history]
      }));
    }
  };

  const selectPatient = (patientId) => {
    const patient = patients.find(p => p.id === patientId);
    setSelectedPatient(patient);
  };

  const filteredPatients = patients.filter(p =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.condition.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="app">
      <div className="header">
        <h1>🏥 CCCARE Exercise Tracking System</h1>
        <p className="subtitle">Real-time Patient Monitoring & Exercise Analytics</p>
      </div>

      <div className="main-container">
        {/* Left Panel */}
        <div className="left-panel">
          {/* Session Control */}
          <div className="card">
            <h2>⚡ Session Control</h2>
            <div className="session-buttons">
              <button
                className="btn btn-start"
                onClick={startSession}
                disabled={sessionActive}
              >
                ▶️ Start Session
              </button>
              <button
                className="btn btn-stop"
                onClick={stopSession}
                disabled={!sessionActive}
              >
                ⏹️ Stop Session
              </button>
            </div>
            <div className={`session-status ${sessionActive ? 'status-active' : 'status-inactive'}`}>
              {sessionActive ? '🟢 Session Active' : '🔴 Session Inactive'}
            </div>
          </div>

          {/* Patient Search */}
          <div className="card">
            <h2>📋 Attendance List</h2>
            <input
              type="text"
              className="search-input"
              placeholder="Search patients by name or condition..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <div className="patient-list">
              {filteredPatients.map(patient => (
                <div
                  key={patient.id}
                  className={`patient-item ${selectedPatient?.id === patient.id ? 'selected' : ''}`}
                  onClick={() => selectPatient(patient.id)}
                >
                  <div className="patient-name">{patient.name}</div>
                  <div className="patient-info">Age: {patient.age} | {patient.condition}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="right-panel">
          {!selectedPatient ? (
            <div className="card">
              <div className="welcome-message">
                <h2>Welcome to CCCARE Tracker</h2>
                <p>Select a patient from the attendance list to begin monitoring</p>
              </div>
            </div>
          ) : (
            <>
              <div className="card">
                <div className="patient-header">
                  <h2>{selectedPatient.name}</h2>
                  <div className="patient-meta">
                    <span>👤 Age: {selectedPatient.age}</span>
                    <span>🏥 {selectedPatient.condition}</span>
                    <span>🩸 Blood Type: {selectedPatient.medicalRecords.bloodType}</span>
                  </div>
                </div>

                {/* Vitals */}
                <div className="vitals-grid">
                  <div className="vital-card heart-rate">
                    <div className="vital-label">❤️ Heart Rate</div>
                    <div className="vital-value">
                      <span>{heartRate}</span>
                      <span className="vital-unit">bpm</span>
                    </div>
                    {showHeartRateAlert && (
                      <div className="alert">
                        ⚠️ Above Target ({Math.round((220 - selectedPatient.age) * 0.7)} bpm)
                      </div>
                    )}
                  </div>

                  <div className="vital-card breathing">
                    <div className="vital-label">🫁 Breathing Rate</div>
                    <div className="vital-value">
                      <span>{breathing}</span>
                      <span className="vital-unit">/min</span>
                    </div>
                  </div>

                  <div className="vital-card oxygen">
                    <div className="vital-label">💧 Oxygen Level</div>
                    <div className="vital-value">
                      <span>{oxygen}</span>
                      <span className="vital-unit">%</span>
                    </div>
                  </div>
                </div>

                {/* Medical Records */}
                <div className="medical-section">
                  <h3>📄 Medical Records</h3>
                  <div className="record-item">
                    <div className="record-label">Diagnosis</div>
                    <div className="record-value">{selectedPatient.medicalRecords.diagnosis}</div>
                  </div>
                  <div className="record-item">
                    <div className="record-label">Current Medications</div>
                    <div className="record-value">{selectedPatient.medicalRecords.medications.join(', ')}</div>
                  </div>
                  <div className="record-item">
                    <div className="record-label">Allergies</div>
                    <div className="record-value">{selectedPatient.medicalRecords.allergies.join(', ')}</div>
                  </div>
                  <div className="record-item">
                    <div className="record-label">Emergency Contact</div>
                    <div className="record-value">{selectedPatient.medicalRecords.emergencyContact}</div>
                  </div>
                </div>

                {/* Training History */}
                <div className="medical-section">
                  <h3>📊 Training History & Feedback</h3>
                  {selectedPatient.history.length === 0 ? (
                    <p style={{textAlign: 'center', color: '#718096', padding: '2rem'}}>
                      No session history available yet
                    </p>
                  ) : (
                    selectedPatient.history.map((session, idx) => (
                      <div key={idx} className="history-item">
                        <div className="history-date">
                          {session.date.toLocaleDateString('en-US', {
                            weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
                          })}
                        </div>
                        <div className="history-stats">
                          <div><strong>Duration:</strong> {session.duration}</div>
                          <div><strong>Avg HR:</strong> {session.avgHeartRate} bpm</div>
                          <div><strong>Max HR:</strong> {session.maxHeartRate} bpm</div>
                          <div><strong>Avg O₂:</strong> {session.avgOxygen}%</div>
                        </div>
                        <div><strong>Exercises:</strong> {session.exerciseCompleted}</div>
                        <div className="training-effect">🎯 Training Effect: {session.trainingEffect}/5.0</div>
                        {session.notes && (
                          <div style={{marginTop: '0.5rem', fontStyle: 'italic', color: '#4a5568'}}>
                            💬 {session.notes}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {sessionActive && (
                <div className="card">
                  <h2>📹 AI Exercise Monitoring</h2>
                  
                  <div className="webcam-container">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    <div className="webcam-placeholder" style={{display: 'none'}}>
                      📹 Camera activated
                    </div>
                    <div className="webcam-overlay">
                      <div className="recording-indicator"></div>
                      Live Monitoring
                    </div>
                  </div>

                  <div className="exercise-metrics">
                    <div className="metric-box">
                      <div className="metric-label">Rep Count</div>
                      <div className="metric-value">{repCount}</div>
                    </div>

                    <div className="metric-box">
                      <div className="metric-label">Current Angle</div>
                      <div className="metric-value">{currentAngle}</div>
                      <div className={`posture-status ${postureStatus.good ? 'posture-good' : 'posture-adjust'}`}>
                        {postureStatus.text}
                      </div>
                    </div>

                    <div className="metric-box">
                      <div className="metric-label">Exercise Quality</div>
                      <div className="metric-value">{exerciseQuality}</div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
