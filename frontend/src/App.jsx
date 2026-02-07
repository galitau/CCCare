import './App.css';
import useAppState from './hooks/useAppState';
import AppView from './components/AppView';

export default function App() {
  const state = useAppState();
  return <AppView {...state} />;

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