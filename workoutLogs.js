import { MongoClient } from "mongodb";

const WORKOUT_DB_NAME = "Users";
const WORKOUT_COLLECTION = "workout_logs";

export async function registerWorkoutLogs(app, mongoUri) {
  const client = new MongoClient(mongoUri);
  await client.connect();
  const db = client.db(WORKOUT_DB_NAME);
  const workoutCol = db.collection(WORKOUT_COLLECTION);
  console.log("✅ Mongo connected ->", WORKOUT_DB_NAME, WORKOUT_COLLECTION);

  app.get("/api/workout-logs", async (req, res) => {
    const patient = String(req.query.patient || "").trim();
    const date = String(req.query.date || "").trim();
    if (!patient) {
      res.json([]);
      return;
    }

    const query = { patient_name: patient };
    if (date) query.date = date;

    const logs = await workoutCol
      .find(query)
      .sort({ last_sync: -1 })
      .toArray();

    res.json(logs);
  });
}
