import WebSocket from 'ws';

const url = 'ws://localhost:5176';
console.log('Connecting to', url);
const ws = new WebSocket(url);

ws.on('open', () => {
  console.log('Connected to backend WebSocket');
});

ws.on('message', (data) => {
  try {
    const msg = JSON.parse(data.toString());
    console.log('MSG:', JSON.stringify(msg));
  } catch (e) {
    console.log('RAW:', data.toString());
  }
});

ws.on('close', () => {
  console.log('Connection closed');
});

ws.on('error', (err) => {
  console.error('WS error:', err.message);
});

// Exit after 20s
setTimeout(() => {
  console.log('Exiting test client');
  ws.close();
  process.exit(0);
}, 20000);
