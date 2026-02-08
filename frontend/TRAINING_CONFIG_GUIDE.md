# Training Effect Calculation Configuration Guide

The training effect score (0-5) is calculated based on three weighted factors in the `useAppState.js` hook. You can easily adjust all parameters in the `TRAINING_CONFIG` object.

## Configuration Parameters

### Heart Rate Zone (% of Max Heart Rate)
```javascript
heartRateZone: {
  min: 0.60,        // 60% - Minimum safe heart rate
  target: { 
    min: 0.70,      // 70% - Lower bound of target zone
    max: 0.85       // 85% - Upper bound of target zone
  },
  max: 1.0          // 100% - Maximum safe heart rate
}
```

**To adjust:** Change the decimal values (0.60, 0.70, 0.85, 1.0) to different percentages. For example:
- More conservative: `min: 0.65, target: { min: 0.65, max: 0.75 }`
- More aggressive: `min: 0.55, target: { min: 0.75, max: 0.90 }`

### Rep Goals by Exercise
```javascript
repGoals: {
  'Arm Raises': 12,
  'Leg Lifts': 12,
  'Squats': 10,
  'Side Bends': 12,
  'Knee Raises': 12,
  'Shoulder Rotations': 15,
  'Ankle Circles': 15
}
```

**To adjust:** Change the number for each exercise. Higher numbers = harder goals.

### Quality Weights
```javascript
qualityWeights: {
  heartRateAdherence: 0.4,     // 40% - Time in target HR zone
  repCompletion: 0.35,          // 35% - Reps completed vs goal
  exerciseQuality: 0.25         // 25% - Fewer corrections = higher quality
}
```

**To adjust:** Change the decimal values. They must sum to 1.0 for proper scaling.

## Calculation Formula

```
Training Effect = (
  (HR adherence % × 0.4) + 
  (Reps completed / Reps goal × 0.35) + 
  (Quality score × 0.25)
) × 5
```

Where:
- **HR adherence %** = Time spent in target heart rate zone (0-1)
- **Rep completion** = Actual reps completed / target reps goal (0-1, capped at 1.0)
- **Quality score** = 1.0 - (corrections per rep × 0.5), minimum 0

## Session Metrics Collected

- **Average Heart Rate**: Calculated from all HR readings during session
- **Max Heart Rate**: Peak HR during session
- **Rep Data**: Reps completed per exercise vs goal
- **Corrections**: Count of form corrections given during session

## Example Scenarios

### Scenario 1: Focus on Heart Rate Control (Senior Patient)
```javascript
heartRateZone: { min: 0.50, target: { min: 0.60, max: 0.75 }, max: 0.85 },
qualityWeights: { heartRateAdherence: 0.50, repCompletion: 0.30, exerciseQuality: 0.20 }
```

### Scenario 2: Focus on Rep Completion (Rehab Patient)
```javascript
repGoals: {
  'Arm Raises': 8,      // Lower goals
  'Leg Lifts': 8,
  'Squats': 6,
  // ... etc
},
qualityWeights: { heartRateAdherence: 0.30, repCompletion: 0.50, exerciseQuality: 0.20 }
```

### Scenario 3: Focus on Form Quality (Strength Training)
```javascript
qualityWeights: { heartRateAdherence: 0.35, repCompletion: 0.30, exerciseQuality: 0.35 }
```

## How to Modify

1. Open `frontend/src/hooks/useAppState.js`
2. Find the `TRAINING_CONFIG` object (around line 107)
3. Adjust values as needed
4. Save the file - changes apply on next session

The system is production-ready and all calculations are done in real-time during sessions.
