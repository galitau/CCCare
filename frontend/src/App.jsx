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
  
  // Current exercise state
  const [currentExercise, setCurrentExercise] = useState('--');
  const [currentReps, setCurrentReps] = useState(0);
  const [currentFeedback, setCurrentFeedback] = useState([]);
  
  // Session history (exercises completed this session)
  const [sessionExercises, setSessionExercises] = useState([]);
  const [sessionNumber, setSessionNumber] = useState(1);
  
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

  // Update session number when patient is selected
  useEffect(() => {
    if (selectedPatient) {
      setSessionNumber(selectedPatient.history.length + 1);
    }
  }, [selectedPatient]);

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

  const generateFeedback = (angle) => {
    const feedback = [];
    if (angle < 60) {
      feedback.push("⚠️ Increase range of motion - raise arm higher");
    }
    if (angle > 120) {
      feedback.push("⚠️ Don't overextend - reduce range slightly");
    }
    if (angle >= 60 && angle <= 120) {
      feedback.push("✅ Excellent form - maintain this position");
    }
    
    // Random additional feedback
    if (Math.random() > 0.7) {
      const additionalFeedback = [
        "Keep your core engaged",
        "Breathe steadily throughout the movement",
        "Maintain good posture - shoulders back",
        "Control the movement - don't rush"
      ];
      feedback.push(additionalFeedback[Math.floor(Math.random() * additionalFeedback.length)]);
    }
    
    return feedback;
  };

  const startExerciseTracking = () => {
    const exercises = [
      'Arm Raises',
      'Leg Lifts', 
      'Squats',
      'Side Bends',
      'Knee Raises',
      'Shoulder Rotations',
      'Ankle Circles'
    ];
    
    let currentExerciseIndex = 0;
    let currentRepCount = 0;
    let exerciseStartTime = Date.now();
    
    // Set initial exercise
    setCurrentExercise(exercises[0]);
    setCurrentReps(0);
    setCurrentFeedback([]);
    
    // Add initial exercise to the list immediately (with 0 reps)
    setSessionExercises([{
      name: exercises[0],
      reps: 0,
      feedback: ["Starting exercise..."],
      duration: 0,
      completedAt: new Date(),
      isActive: true
    }]);
    
    exerciseIntervalRef.current = setInterval(() => {
      if (!sessionActive) return;

      // Simulate rep counting (in real implementation, this comes from webcam AI)
      if (Math.random() > 0.6) {
        currentRepCount++;
        setCurrentReps(currentRepCount);
        
        // Generate feedback based on simulated angle
        const angle = Math.floor(Math.random() * 180);
        const feedback = generateFeedback(angle);
        setCurrentFeedback(feedback);
        
        // Update the current (first) exercise in the list with new rep count and feedback
        setSessionExercises(prev => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[0] = {
              ...updated[0],
              reps: currentRepCount,
              feedback: feedback,
              isActive: true
            };
          }
          return updated;
        });
        
        // After 8-12 reps, move to next exercise
        const targetReps = 8 + Math.floor(Math.random() * 5); // 8-12 reps
        if (currentRepCount >= targetReps) {
          // Mark current exercise as completed
          setSessionExercises(prev => {
            const updated = [...prev];
            if (updated.length > 0) {
              updated[0] = {
                ...updated[0],
                isActive: false,
                duration: Math.round((Date.now() - exerciseStartTime) / 1000)
              };
            }
            return updated;
          });
          
          // Move to next exercise
          currentExerciseIndex = (currentExerciseIndex + 1) % exercises.length;
          currentRepCount = 0;
          exerciseStartTime = Date.now();
          
          setCurrentExercise(exercises[currentExerciseIndex]);
          setCurrentReps(0);
          setCurrentFeedback(["Starting new exercise..."]);
          
          // Add new exercise to the top of the list
          setSessionExercises(prev => [{
            name: exercises[currentExerciseIndex],
            reps: 0,
            feedback: ["Starting exercise..."],
            duration: 0,
            completedAt: new Date(),
            isActive: true
          }, ...prev]);
        }
      } else {
        // Update feedback even when not counting reps
        const angle = Math.floor(Math.random() * 180);
        const feedback = generateFeedback(angle);
        setCurrentFeedback(feedback);
        
        // Update current exercise feedback
        setSessionExercises(prev => {
          const updated = [...prev];
          if (updated.length > 0 && updated[0].isActive) {
            updated[0] = {
              ...updated[0],
              feedback: feedback
            };
          }
          return updated;
        });
      }
    }, 2000);
  };

  const stopExerciseTracking = () => {
    if (exerciseIntervalRef.current) {
      clearInterval(exerciseIntervalRef.current);
      exerciseIntervalRef.current = null;
    }
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
    setSessionExercises([]);
    setCurrentExercise('--');
    setCurrentReps(0);
    setCurrentFeedback([]);
    
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
      // Create summary of exercises completed (filter out the active one with 0 reps if exists)
      const completedExercises = sessionExercises.filter(ex => !ex.isActive || ex.reps > 0);
      const exerciseSummary = completedExercises
        .map(ex => `${ex.name} (${ex.reps} reps)`)
        .join(', ');
      
      const newSession = {
        date: new Date(),
        duration: "30 minutes",
        avgHeartRate: 78,
        maxHeartRate: 95,
        avgOxygen: 97,
        exerciseCompleted: exerciseSummary || "Session incomplete",
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
      
      // Increment session number for next session
      setSessionNumber(prev => prev + 1);
      
      // Reset current exercise state
      setCurrentExercise('--');
      setCurrentReps(0);
      setCurrentFeedback([]);
      setSessionExercises([]);
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
                  </div>
                </div>

                {/* Vitals - 3 cards only */}
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

                {/* Current Session Section */}
                <div className="medical-section">
                  <h3>🏋️ Current Session #{sessionNumber}</h3>
                  
                  {!sessionActive ? (
                    <p style={{textAlign: 'center', color: '#718096', padding: '2rem'}}>
                      No active session. Click "Start Session" to begin.
                    </p>
                  ) : (
                    <>
                      {sessionExercises.length === 0 ? (
                        <p style={{textAlign: 'center', color: '#718096', padding: '2rem'}}>
                          Starting session...
                        </p>
                      ) : (
                        sessionExercises.map((exercise, idx) => (
                          <div key={idx} className={`completed-exercise-item ${exercise.isActive ? 'active-exercise' : ''}`}>
                            <div className="exercise-header-row">
                              <span className="exercise-number">
                                {exercise.isActive ? '▶' : `#${sessionExercises.length - idx}`}
                              </span>
                              <span className="exercise-name">{exercise.name}</span>
                              <span className="exercise-reps">{exercise.reps} reps</span>
                            </div>
                            {exercise.feedback && exercise.feedback.length > 0 && (
                              <div className="exercise-feedback">
                                <strong>Feedback:</strong>
                                {exercise.feedback.map((fb, fbIdx) => (
                                  <div key={fbIdx} className="feedback-text">• {fb}</div>
                                ))}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </>
                  )}
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}