require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const app = express();

app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/wheel_inspection';  

// Connect to MongoDB (either local or Atlas)
mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);  // Exit the process if MongoDB connection fails
  });

// Define Report schema and model
const reportSchema = new mongoose.Schema({
  timestamp: { type: Date, required: true },
  status: { type: String, required: true },
  recommendation: { type: String, default: '' },
  image_path: { type: String, required: true },
  name: { type: String, default: 'Untitled Report' },
});

const Report = mongoose.model('Report', reportSchema);

// WebSocket server
const server = app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});

// Initialize WebSocket server
const wss = new WebSocket.Server({ server });
wss.on('connection', (ws) => {
  console.log('New WebSocket client connected');
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// WebSocket message handling - Broadcasting new reports
const broadcastNewReport = (report) => {
  const message = JSON.stringify({ type: 'new_report', data: report });
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
};

// API routes

// Test route to check if backend is working
app.get('/test', (req, res) => {
  res.send('Backend is working!');
});

// Create new report (POST)
app.post('/api/reports', async (req, res) => {
  try {
    // Automatically generate the timestamp
    const timestamp = new Date().toISOString();  // Get current date and time in ISO format
    
    // Set report name based on timestamp
    const reportName = `Inspection ${timestamp}`;

    // Automatically generate image path (you could modify this based on your actual image saving logic)
    const imagePath = `uploads/${reportName.replace(/[:.]/g, '-')}.jpg`;

    // Generate the status based on whether the image is flawed or not
    const status = req.body.status || 'NO FLAW';  // You might update this later based on your image analysis logic
    const recommendation = status === 'FLAW DETECTED' ? 'For Repair/Replacement' : 'For Constant Monitoring';

    // Create the new report with the generated data
    const reportData = {
      timestamp: new Date(),
      status,
      recommendation,
      image_path: imagePath,
      name: reportName
    };

    const report = new Report(reportData);
    await report.save();

    // Broadcast the new report to all connected WebSocket clients
    broadcastNewReport(report);

    // Respond with the created report
    res.status(201).json(report);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Get all reports (GET)
app.get('/api/reports', async (req, res) => {
  try {
    const reports = await Report.find().sort({ timestamp: -1 });
    res.json(reports);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get report by ID (GET)
app.get('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findById(req.params.id);
    if (!report) return res.status(404).json({ error: 'Report not found' });
    res.json(report);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update report by ID (PUT)
app.put('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!report) return res.status(404).json({ error: 'Report not found' });
    res.json(report);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Delete report by ID (DELETE)
app.delete('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findByIdAndDelete(req.params.id);
    if (!report) return res.status(404).json({ error: 'Report not found' });
    res.json({ message: 'Report deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Gracefully shut down the server
process.on('SIGINT', () => {
  console.log('Shutting down WebSocket server...');
  wss.close(() => {
    console.log('WebSocket server closed');
    server.close(() => {
      console.log('HTTP server closed');
      process.exit(0);  // Exit the process after closing servers
    });
  });
});
