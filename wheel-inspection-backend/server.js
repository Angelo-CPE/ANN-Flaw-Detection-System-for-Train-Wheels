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
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();

// Middleware
app.use(cors());
app.use(helmet());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`Incoming ${req.method} request to ${req.path}`);
  console.log('Headers:', req.headers['content-type']);
  next();
});

// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100
});
app.use(limiter);

const cleanupDuplicates = async () => {
  const allReports = await Report.find().sort({ timestamp: -1 });
  const seen = new Set();
  const toDelete = [];

  for (const r of allReports) {
    const date = new Date(r.timestamp).toISOString().slice(0, 10);
    const key = `${r.trainNumber}-${r.compartmentNumber}-${r.wheelNumber}-${date}`;
    if (seen.has(key)) {
      toDelete.push(r);
    } else {
      seen.add(key);
    }
  }

  for (const r of toDelete) {
    const filePath = path.join(__dirname, r.image_path.replace('/uploads/', 'uploads/'));
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    await Report.deleteOne({ _id: r._id });
    console.log(`🗑 Deleted: ${r.name}`);
  }

  console.log(`✅ Removed ${toDelete.length} duplicates.`);
};


const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://pdTeam39:t39@cluster0.khfnesv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

// Connect to MongoDB
mongoose.connect(MONGODB_URI, { 
  useNewUrlParser: true, 
  useUnifiedTopology: true,
  dbName: 'wheel_inspection'
  
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

cleanupDuplicates();

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
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Report Schema and Model
const reportSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  status: { type: String, required: true, enum: ['NO FLAW', 'FLAW DETECTED'] },
  recommendation: { type: String },
  image_path: { type: String, required: true },
  name: { type: String },
  trainNumber: { type: String, required: true },
  compartmentNumber: { type: String, required: true },
  wheelNumber: { type: String, required: true },
  wheel_diameter: { type: String, required: true }
});

reportSchema.pre('save', function(next) {
  this.recommendation = this.status === 'FLAW DETECTED' 
    ? 'For Repair/Replacement' 
    : 'For Constant Monitoring';
  this.name = `Flaw Inspection T${this.trainNumber}-C${this.compartmentNumber}-W${this.wheelNumber}`;
  next();
});

const Report = mongoose.model('Report', reportSchema);

// WebSocket Server
const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  console.log('New WebSocket client connected:', req.socket.remoteAddress);

  ws.on('error', (err) => {
    console.error('WebSocket error:', err.message);
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});


const broadcastReportUpdate = (action, data) => {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ 
        type: 'report_update', 
        action,
        data
      }));
    }
  });
};

// API Routes
app.get('/test', (req, res) => {
  res.send('Backend is working!');
});

app.post('/api/reports', upload.single('image'), async (req, res) => {
  try {
    console.log('Request Body:', req.body);
    console.log('Uploaded File:', req.file);

    const { 
      trainNumber, 
      compartmentNumber, 
      wheelNumber, 
      status,
      wheel_diameter 
    } = req.body;
    
    if (!trainNumber || !compartmentNumber || !wheelNumber || !wheel_diameter || !req.file) {
      if (req.file) fs.unlinkSync(req.file.path);
      return res.status(400).json({ 
        error: 'Missing required fields',
        required: ['trainNumber', 'compartmentNumber', 'wheelNumber', 'wheel_diameter', 'image']
      });
    }

    const today = new Date().toISOString().slice(0, 10);
      const startOfDay = new Date(`${today}T00:00:00.000Z`);
      const endOfDay = new Date(`${today}T23:59:59.999Z`);

      const existing = await Report.findOne({
        trainNumber,
        compartmentNumber,
        wheelNumber,
        timestamp: { $gte: startOfDay, $lte: endOfDay }
      });

      if (existing) {
        const imagePath = path.join(__dirname, existing.image_path.replace('/uploads/', 'uploads/'));
        if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);
        await Report.deleteOne({ _id: existing._id });
      }

      const report = new Report({
        trainNumber,
        compartmentNumber,
        wheelNumber,
        wheel_diameter,
        status,
        recommendation,
        name,
        image_path: `/uploads/${req.file.filename}`
      });


    await report.save();


    broadcastReportUpdate('created', report);
    
    res.status(201).json(report);
  } catch (err) {
    if (req.file) fs.unlinkSync(req.file.path);
    res.status(400).json({ 
      error: err.message,
      details: 'Ensure all fields are sent as form-data with correct content types'
    });
  }
});

app.get('/api/reports', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;
    
    const reports = await Report.find()
      .sort({ timestamp: -1 })
      .skip(skip)
      .limit(limit);
      
    const total = await Report.countDocuments();
    
    res.json({
      data: reports,
      meta: {
        total,
        page,
        pages: Math.ceil(total / limit)
      }
    });
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
    const report = await Report.findByIdAndUpdate(
      req.params.id, 
      req.body, 
      { new: true, runValidators: true }
    );
    
    if (!report) return res.status(404).json({ error: 'Report not found' });
    
    broadcastReportUpdate('updated', report);
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
    
    broadcastReportUpdate('deleted', report._id);
    res.json({ message: 'Report deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// New Endpoints
app.get('/api/trains', async (req, res) => {
  try {
    const trains = await Report.distinct('trainNumber');
    res.json(trains);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/compartments/:trainNumber', async (req, res) => {
  try {
    const compartments = await Report.distinct('compartmentNumber', {
      trainNumber: req.params.trainNumber
    });
    res.json(compartments);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Error Handling
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ 
      error: 'File upload error',
      message: err.message,
      code: err.code
    });
  }
  
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message
  });
});

process.on('SIGINT', () => {
  wss.close(() => {
    server.close(() => {
      process.exit(0);
    });
  });
});
