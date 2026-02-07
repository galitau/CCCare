import { useState, useEffect, useRef } from 'react';

export default function useAppState() {
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [heartRate, setHeartRate] = useState('--');
  const [breathing, setBreathing] = useState('--');
  const [oxygen, setOxygen] = useState('--');
  const [showHeartRateAlert, setShowHeartRateAlert] = useState(false);

  const [currentExercise, setCurrentExercise] = useState('--');
  const [currentReps, setCurrentReps] = useState(0);
  const [currentFeedback, setCurrentFeedback] = useState([]);

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
  const wsRef = useRef(null);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    return () => {
      if (vitalsIntervalRef.current) clearInterval(vitalsIntervalRef.current);
      if (exerciseIntervalRef.current) clearInterval(exerciseIntervalRef.current);
      if (webcamStreamRef.current) {
        webcamStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (wsRef.current) {
        try { wsRef.current.close(); } catch (e) {}
        wsRef.current = null;
      }
    };
  }, []);

  // WebSocket connection to backend for live vitals (hr / spo2)
  useEffect(() => {
    let reconnectTimer = null;

    function connectWs() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const host = window.location.hostname || 'localhost';
      const url = `${protocol}://${host}:5176`;

      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setWsConnected(true);
        };

        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.hr !== undefined) setHeartRate(data.hr);
            if (data.spo2 !== undefined) setOxygen(data.spo2);
            if (data.br !== undefined) setBreathing(data.br);
          } catch (e) {
            // ignore non-JSON messages
          }
        };

        ws.onclose = () => {
          setWsConnected(false);
          // attempt reconnect
          if (reconnectTimer) clearTimeout(reconnectTimer);
          reconnectTimer = setTimeout(connectWs, 2000);
        };

        ws.onerror = () => {
          try { ws.close(); } catch (e) {}
        };
      } catch (e) {
        setWsConnected(false);
        reconnectTimer = setTimeout(connectWs, 2000);
      }
    }

    connectWs();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        try { wsRef.current.close(); } catch (e) {}
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (selectedPatient) {
      setSessionNumber(selectedPatient.history.length + 1);
    }
  }, [selectedPatient]);

  const startVitalsMonitoring = () => {
    // vitals are provided by the backend WebSocket; no local generator.
    // Ensure any previous generator is cleared.
    if (vitalsIntervalRef.current) {
      clearInterval(vitalsIntervalRef.current);
      vitalsIntervalRef.current = null;
    }
    // Reset placeholders until server data arrives
    setHeartRate('--');
    setBreathing('--');
    setOxygen('--');
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
    
    setCurrentExercise(exercises[0]);
    setCurrentReps(0);
    setCurrentFeedback([]);
    
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

      if (Math.random() > 0.6) {
        currentRepCount++;
        setCurrentReps(currentRepCount);
        
        const angle = Math.floor(Math.random() * 180);
        const feedback = generateFeedback(angle);
        setCurrentFeedback(feedback);
        
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
        
        const targetReps = 8 + Math.floor(Math.random() * 5);
        if (currentRepCount >= targetReps) {
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
          
          currentExerciseIndex = (currentExerciseIndex + 1) % exercises.length;
          currentRepCount = 0;
          exerciseStartTime = Date.now();
          
          setCurrentExercise(exercises[currentExerciseIndex]);
          setCurrentReps(0);
          setCurrentFeedback(["Starting new exercise..."]);
          
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
        const angle = Math.floor(Math.random() * 180);
        const feedback = generateFeedback(angle);
        setCurrentFeedback(feedback);
        
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
      const completedExercises = sessionExercises.filter(ex => !ex.isActive || ex.reps > 0);
      const exerciseSummary = completedExercises
        .map(ex => `${ex.name} (${ex.reps} reps)`)
        .join(', ');
      
      const newSession = {
        date: new Date(),
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
      
      setSessionNumber(prev => prev + 1);
      
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

  return {
    selectedPatient,
    sessionActive,
    searchTerm,
    setSearchTerm,
    heartRate,
    breathing,
    oxygen,
    showHeartRateAlert,
    currentExercise,
    currentReps,
    currentFeedback,
    sessionExercises,
    sessionNumber,
    patients,
    setPatients,
    startSession,
    stopSession,
    selectPatient,
    filteredPatients,
    videoRef
  };
}
