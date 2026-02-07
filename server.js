import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import { SerialPort } from "serialport";
import { ReadlineParser } from "@serialport/parser-readline";
import { MongoClient } from "mongodb";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 5176;

// --------------------
// MongoDB
// --------------------
const MONGO_URI = process.env.MONGO_URI;
const DB_NAME = "Jisara_W";
const COLLECTION = "OxygenRate";

let vitalsCol;

async function connectMongo() {
  const client = new MongoClient(MONGO_URI);
  await client.connect();
  const db = client.db(DB_NAME);
  vitalsCol = db.collection(COLLECTION);
  console.log("✅ Mongo connected ->", DB_NAME, COLLECTION);
}

await connectMongo();

// --------------------
// WebSocket
// --------------------
const server = app.listen(PORT, () => {
  console.log(`✅ Backend running on http://localhost:${PORT}`);
});

const wss = new WebSocketServer({ server });

function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === 1) client.send(msg);
  });
}

// --------------------
// Serial (Auto-Reconnect)
// --------------------
const SERIAL_PORT = "COM5";
const BAUD_RATE = 115200;

let port;
let parser;

function connectSerial() {
  console.log("🔌 Trying to connect to", SERIAL_PORT);

  port = new SerialPort({
    path: SERIAL_PORT,
    baudRate: BAUD_RATE,
    autoOpen: false
  });

  port.open((err) => {
    if (err) {
      console.log("❌ Open error:", err.message);
      setTimeout(connectSerial, 2000);
      return;
    }

    console.log("✅ Serial connected");

    parser = port.pipe(new ReadlineParser({ delimiter: "\n" }));

    parser.on("data", async (line) => {
      const raw = String(line).trim().replace(/\r$/, "");
      if (!raw) return;

      console.log("SERIAL RAW:", raw);

      let msg;
      try {
        msg = JSON.parse(raw);
      } catch {
        return; // ignore non-JSON
      }

      broadcast(msg);

      // Only store vitals
      if (msg.spo2 !== undefined && msg.hr !== undefined) {
        if (msg.spo2 >= 0) {
          try {
            await vitalsCol.insertOne({
              spo2: msg.spo2,
              hr: msg.hr,
              ts: new Date()
            });
            console.log("✅ Inserted into Mongo");
          } catch (e) {
            console.log("❌ Mongo error:", e.message);
          }
        }
      }
    });
  });

  port.on("close", () => {
    console.log("⚠️ Serial port closed — reconnecting...");
    setTimeout(connectSerial, 2000);
  });

  port.on("error", (err) => {
    console.log("❌ Serial error:", err.message);
  });
}

connectSerial();

// --------------------
// API Endpoint
// --------------------
app.get("/api/latest", async (req, res) => {
  const latest = await vitalsCol
    .find({})
    .sort({ ts: -1 })
    .limit(1)
    .toArray();

  res.json(latest[0] || null);
});
