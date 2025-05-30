require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const crypto = require('crypto');

const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://pdTeam39:t39@cluster0.khfnesv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

// Connect to MongoDB
mongoose.connect(MONGODB_URI, { 
  useNewUrlParser: true, 
  useUnifiedTopology: true,
  dbName: 'wheel_inspection' // Specify your database name
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

// Configure storage for uploaded images
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const uniqueSuffix = Date.now() + '-' + crypto.randomBytes(4).toString('hex');
    cb(null, uniqueSuffix + ext);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Define Report schema and model
const reportSchema = new mongoose.Schema({
  timestamp: { type: Date, required: true },
  name: { type: String, default: 'Untitled Report' },
  train_number: { type: Number, required: true },
  compartment_number: { type: Number, required: true },
  wheel_number: { type: Number, required: true },
  status: { type: String, required: true },
  recommendation: { type: String, default: '' },
  image_path: { type: String, required: true },
  wheel_diameter: { type: Number, required: true }
});


// Auto-set recommendation based on status
reportSchema.pre('save', function(next) {
  this.recommendation = this.status === 'FLAW DETECTED' 
    ? 'For Repair/Replacement' 
    : 'For Constant Monitoring';
  this.name = `Flaw Inspection Train ${this.train_number}-C${this.compartment_number}-W${this.wheel_number}`;
  next();
});

const Report = mongoose.model('Report', reportSchema);

// WebSocket server
const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});

const wss = new WebSocket.Server({ server });
wss.on('connection', (ws) => {
  console.log('New WebSocket client connected');
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

const broadcastNewReport = (report) => {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'new_report', data: report }));
    }
  });
};

// API routes
app.get('/test', (req, res) => {
  res.send('Backend is working!');
});
 
app.post('/api/reports', upload.single('image'), async (req, res) => {
  try {
    console.log('Incoming report:', req.body);  // 👈 ADD THIS LINE

    const { train_number, compartment_number, wheel_number, status, image_path, recommendation, name, notes, wheel_diameter } = req.body;
    const report = new Report({
      timestamp: new Date(),
      train_number,
      compartment_number,
      wheel_number,
      status,
      image_path,
      recommendation,
      name,
      notes,
      wheel_diameter
    });

    await report.save();
    broadcastNewReport(report);
    res.status(201).json(report);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.get('/api/reports', async (req, res) => {
  try {
    const reports = await Report.find().sort({ timestamp: -1 });
    res.json(reports);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findById(req.params.id);
    if (!report) return res.status(404).json({ error: 'Report not found' });
    res.json(report);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findByIdAndUpdate(req.params.id, req.body, { 
      new: true,
      runValidators: true 
    });
    if (!report) return res.status(404).json({ error: 'Report not found' });
    res.json(report);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.delete('/api/reports/:id', async (req, res) => {
  try {
    const report = await Report.findByIdAndDelete(req.params.id);
    if (!report) return res.status(404).json({ error: 'Report not found' });
    
    const imagePath = path.join(__dirname, report.image_path.replace('/uploads/', 'uploads/'));
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }
    
    res.json({ message: 'Report deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: 'File upload error: ' + err.message });
  }
  res.status(500).json({ error: err.message });
});

process.on('SIGINT', () => {
  wss.close(() => {
    server.close(() => {
      process.exit(0);
    });
  });
});