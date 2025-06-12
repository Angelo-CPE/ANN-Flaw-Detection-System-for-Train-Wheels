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
  const bcrypt = require('bcryptjs');
  const jwt = require('jsonwebtoken');
  const nodemailer = require('nodemailer');
  const { body, validationResult } = require('express-validator');

  const app = express();

  app.use(cors());
  app.use(helmet());
  app.use(bodyParser.json());
  app.use(bodyParser.urlencoded({ extended: true }));
  app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

  app.use((req, res, next) => {
    console.log(`Incoming ${req.method} request to ${req.path}`);
    next();
  });

  const limiter = rateLimit({
    windowMs: 60 * 1000,
    max: 100
  });
  app.use(limiter);

  const PORT = process.env.PORT || 5000;
  const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://pdTeam39:t39@cluster0.khfnesv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
  const JWT_SECRET = process.env.JWT_SECRET;
  const EMAIL_USER = process.env.EMAIL_USER;
  const EMAIL_PASS = process.env.EMAIL_PASS;

  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: EMAIL_USER,
      pass: EMAIL_PASS
    }
  });

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

  const userSchema = new mongoose.Schema({
    email: { 
      isActive: { type: Boolean, default: true },
      type: String, 
      required: true, 
      unique: true,
      validate: {
        validator: function(v) {
          return /@tip\.edu\.ph$/.test(v);
        },
        message: props => `${props.value} is not a valid TIP email address!`
      }
    },
    password: { 
      type: String, 
      required: true, 
      select: false,
      validate: {
        validator: function(v) {
          return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(v);
        },
        message: props => 
          'Password must contain at least 8 characters, one uppercase, one lowercase, one number and one special character'
      }
    },
    name: { type: String, required: true },
    role: { type: String, enum: ['user', 'admin'], default: 'user' },
    createdAt: { type: Date, default: Date.now },

    otp: { type: String },
    otpExpiry: { type: Date },

    resetPasswordToken: String,
    resetPasswordExpire: Date
  });


  // Hash password before saving
  userSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    
    try {
      const salt = await bcrypt.genSalt(10);
      this.password = await bcrypt.hash(this.password, salt);
      next();
    } catch (err) {
      next(err);
    }
  });

  // Method to generate JWT token
  userSchema.methods.getSignedJwtToken = function() {
    return jwt.sign({ id: this._id }, JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRE || '30d'
    });
  };

  // Method to check password
  userSchema.methods.matchPassword = async function(enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
  };

  // Method to generate password reset token
  userSchema.methods.getResetPasswordToken = function() {
    const resetToken = crypto.randomBytes(20).toString('hex');
    
    this.resetPasswordToken = crypto
      .createHash('sha256')
      .update(resetToken)
      .digest('hex');
    
    this.resetPasswordExpire = Date.now() + 10 * 60 * 1000; // 10 minutes
    
    return resetToken;
  };

  const User = mongoose.model('User', userSchema);

  // Report Schema and Model
  const reportSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now },
    status: { 
      type: String, 
      required: true, 
      enum: ['NO FLAW', 'FLAW DETECTED'],
      validate: {
        validator: function(v) {
          return ['NO FLAW', 'FLAW DETECTED'].includes(v);
        },
        message: props => `${props.value} is not a valid status!`
      }
    },
    recommendation: { type: String },
    image_path: { type: String, required: true },
    name: { type: String },
    trainNumber: { type: String, required: true },
    compartmentNumber: { type: String, required: true },
    wheelNumber: { type: String, required: true },
    wheel_diameter: { type: String, required: true },
    createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }
  });

  reportSchema.pre('save', function(next) {
    if (!this.recommendation) {
      this.recommendation = this.status === 'FLAW DETECTED' 
        ? 'For Repair/Replacement' 
        : 'For Constant Monitoring';
    }
    if (!this.name) {
      this.name = `Train ${this.trainNumber} - Compartment ${this.compartmentNumber} - Wheel ${this.wheelNumber}`;
    }
    next();
  });

  const Report = mongoose.model('Report', reportSchema);

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

  const sendTokenResponse = (user, statusCode, res) => {
    const token = user.getSignedJwtToken();

    const options = {
      expires: new Date(Date.now() + Number(process.env.JWT_COOKIE_EXPIRE) * 24 * 60 * 60 * 1000),
      httpOnly: true
    };

    if (process.env.NODE_ENV === 'production') {
      options.secure = true;
    }

    res
      .status(statusCode)
      .cookie('token', token, options)
      .json({
        success: true,
        token,
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role
        }
      });
  };

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

  // Auth Middleware
  const protect = async (req, res, next) => {
    let token;
    
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    
    if (!token) {
      return res.status(401).json({ success: false, error: 'Not authorized to access this route' });
    }
    
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      req.user = await User.findById(decoded.id).select('-password');
      next();
    } catch (err) {
      return res.status(401).json({ success: false, error: 'Not authorized to access this route' });
    }
  };

  const authorize = (...roles) => {
    return (req, res, next) => {
      if (!roles.includes(req.user.role)) {
        return res.status(403).json({ 
          success: false, 
          error: `User role ${req.user.role} is not authorized to access this route`
        });
      }
      next();
    };
  };

  // API Routes

  // Auth Routes 
  app.get('/api/auth/me', protect, async (req, res) => {
    try {
      const user = await User.findById(req.user.id).select('-password');
      res.status(200).json({
        success: true,
        data: user
      });
    } catch (err) {
      res.status(500).json({
        success: false,
        error: 'Server error'
      });
    }
  });
  app.post('/api/auth/register', [
    body('email').isEmail().withMessage('Please include a valid email').custom(value => {
      if (!value.endsWith('@tip.edu.ph')) {
        throw new Error('Only TIP email addresses are allowed for registration');
      }
      return true;
    }),
    body('password')
      .isLength({ min: 8 }).withMessage('Password must be at least 8 characters')
      .matches(/[a-z]/).withMessage('Password must contain at least one lowercase letter')
      .matches(/[A-Z]/).withMessage('Password must contain at least one uppercase letter')
      .matches(/\d/).withMessage('Password must contain at least one number')
      .matches(/[@$!%*?&]/).withMessage('Password must contain at least one special character (@$!%*?&)'),
    body('name').notEmpty().withMessage('Please include a name')
  ], async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    
    try {
      const { email, password, name } = req.body;

      const existingUser = await User.findOne({ email });
      if (existingUser) {
        return res.status(400).json({ success: false, error: 'User already exists' });
      }

      const user = await User.create({
        email,
        password,
        name
      });

      sendTokenResponse(user, 201, res);
    } catch (err) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/auth/login', [
    body('email').isEmail().withMessage('Please include a valid email'),
    body('password').exists().withMessage('Please include a password')
  ], async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    
    try {
      const { email, password } = req.body;
      
      const user = await User.findOne({ email }).select('+password');
      if (!user) {
        return res.status(401).json({ success: false, error: 'Invalid credentials' });
      }
      
      const isMatch = await user.matchPassword(password);
      if (!isMatch) {
        return res.status(401).json({ success: false, error: 'Invalid credentials' });
      }
      
      sendTokenResponse(user, 200, res);
    } catch (err) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/auth/forgotpassword', async (req, res) => {
    const { email } = req.body;
    try {
      const user = await User.findOne({ email });
      if (!user) {
        return res.status(404).json({ error: 'No user with that email' });
      }

      const resetToken = user.getResetPasswordToken();
      await user.save();

      const resetUrl = `https://ann-flaw-detection-system-for-train.onrender.com/resetpassword/${resetToken}`;
      const message = `You requested a password reset. Use the link to reset your password:\n\n${resetUrl}\n\nIf you did not request this, please ignore this email.`;

      await transporter.sendMail({
        to: user.email,
        subject: 'Password Reset Request',
        text: message
      });

      res.json({ message: 'Reset link sent to email' });
    } catch (err) {
      console.error('Forgot password error:', err);
      res.status(500).json({ error: 'Email could not be sent' });
    }
  });

  app.put('/api/auth/resetpassword/:token', async (req, res) => {
    const resetPasswordToken = crypto
      .createHash('sha256')
      .update(req.params.token)
      .digest('hex');

    try {
      const user = await User.findOne({
        resetPasswordToken,
        resetPasswordExpire: { $gt: Date.now() }
      });

      if (!user) {
        return res.status(400).json({ error: 'Invalid or expired token' });
      }

      user.password = req.body.password;
      user.resetPasswordToken = undefined;
      user.resetPasswordExpire = undefined;

      await user.save();

      res.json({ message: 'Password reset successful' });
    } catch (err) {
      console.error('Reset password error:', err);
      res.status(500).json({ error: 'Failed to reset password' });
    }
  });

  // Generate and send OTP
  app.post('/api/auth/request-otp', async (req, res) => {
    const { email } = req.body;
    
    try {
      const user = await User.findOne({ email });
      if (!user) {
        return res.status(404).json({ error: 'No user with this email' });
      }

      // Generate 6-digit OTP
      const otp = Math.floor(100000 + Math.random() * 900000).toString();
      const otpExpiry = Date.now() + 10 * 60 * 1000; // 10 minutes

      user.otp = otp;
      user.otpExpiry = otpExpiry;
      await user.save();

      // Send email
      await transporter.sendMail({
        to: user.email,
        subject: 'Your Password Reset OTP',
        text: `Your OTP is: ${otp}\n\nValid for 10 minutes.`
      });

      res.json({ message: 'OTP sent successfully' });
    } catch (err) {
      console.error('OTP send error:', err);
      res.status(500).json({ error: 'Failed to send OTP' });
    }
  });

  // Verify OTP
  app.post('/api/auth/verify-otp', async (req, res) => {
    const { email, otp } = req.body;
    
    try {
      const user = await User.findOne({ 
        email,
        otp,
        otpExpiry: { $gt: Date.now() }
      });

      if (!user) {
        return res.status(400).json({ error: 'Invalid or expired OTP' });
      }

      // OTP is valid
      res.json({ 
        success: true,
        tempToken: jwt.sign({ email }, JWT_SECRET, { expiresIn: '5m' })
      });
    } catch (err) {
      res.status(500).json({ error: 'OTP verification failed' });
    }
  });

  // Update password after OTP verification
  app.put('/api/auth/update-password', async (req, res) => {
  const { tempToken, password } = req.body;

  try {
    console.log('[DEBUG] tempToken received:', tempToken);
    const decoded = jwt.verify(tempToken, JWT_SECRET);
    console.log('[DEBUG] token payload:', decoded);

    const user = await User.findOne({ email: decoded.email }).select('+password');
    if (!user) {
      return res.status(400).json({ error: 'Invalid token' });
    }

    const isSame = await bcrypt.compare(password, user.password);
    if (isSame) {
      return res.status(400).json({ error: 'New password cannot be the same as the old password' });
    }

    user.password = password;
    user.otp = undefined;
    user.otpExpiry = undefined;
    await user.save();

    res.json({ message: 'Password updated successfully' });
  } catch (err) {
    console.error('[update-password]', err);
    res.status(401).json({ error: 'Token expired or invalid' });
  }
});

//Admin Route (promote/demote)
app.put('/api/admin/users/:id/deactivate', protect, authorize('admin'), async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { isActive: false },
      { new: true }
    ).select('-password');
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ success: true, data: user });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.patch('/api/admin/users/:id/demote', protect, authorize('admin'), async (req, res) => {
  try {
    const user = await User.findById(req.params.id);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (user.role !== 'admin') {
      return res.status(400).json({ error: 'User is not an admin' });
    }

    user.role = 'user';
    await user.save();

    res.json({ success: true, data: user });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.put('/api/admin/users/:id/reactivate', protect, authorize('admin'), async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { isActive: true },
      { new: true }
    ).select('-password');
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ success: true, data: user });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get('/api/admin/users/archived', protect, authorize('admin'), async (req, res) => {
  try {
    const users = await User.find({ isActive: false }).select('-password');
    res.status(200).json({ success: true, data: users });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

  // Report Routes (protected)
  app.get('/api/reports', protect, async (req, res) => {
    try {
      const page = parseInt(req.query.page) || 1;
      const limit = parseInt(req.query.limit) || 20;
      const skip = (page - 1) * limit;
      
      const reports = await Report.find()
        .sort({ timestamp: -1 })
        .skip(skip)
        .limit(limit)
        .populate('createdBy', 'name email');
        
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

  app.get('/api/reports/:id', protect, async (req, res) => {
    try {
      const report = await Report.findById(req.params.id).populate('createdBy', 'name email');
      if (!report) return res.status(404).json({ error: 'Report not found' });
      res.json(report);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  app.post('/api/reports', upload.single('image'), async (req, res) => {
    try {
      const { 
        trainNumber, 
        compartmentNumber, 
        wheelNumber, 
        status,
        recommendation,
        name,
        wheel_diameter 
      } = req.body;
      
      if (!trainNumber || !compartmentNumber || !wheelNumber || 
          !wheel_diameter || !status || !req.file) {
        if (req.file) fs.unlinkSync(req.file.path);
        return res.status(400).json({ 
          error: 'Missing required fields',
          required: ['trainNumber', 'compartmentNumber', 'wheelNumber', 
                    'wheel_diameter', 'status', 'image']
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
        image_path: `/uploads/${req.file.filename}`,
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

  app.put('/api/reports/:id', protect, async (req, res) => {
    try {
      let report = await Report.findById(req.params.id);
      
      if (!report) {
        return res.status(404).json({ error: 'Report not found' });
      }
      
      // Check if user created the report or is admin
      if (report.createdBy.toString() !== req.user.id && req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Not authorized to update this report' });
      }
      
      report = await Report.findByIdAndUpdate(req.params.id, req.body, { 
        new: true, 
        runValidators: true 
      });
      
      broadcastReportUpdate('updated', report);
      res.json(report);
    } catch (err) {
      res.status(400).json({ error: err.message });
    }
  });

  app.patch('/api/admin/users/:id/promote', protect, authorize('admin'), async (req, res) => {
    try {
      const user = await User.findById(req.params.id);

      if (!user) return res.status(404).json({ error: 'User not found' });

      if (user.role === 'admin') {
        return res.status(400).json({ error: 'User is already an admin' });
      }

      user.role = 'admin';
      await user.save();

      res.json({ message: `${user.name} has been promoted to admin.` });
    } catch (err) {
      console.error('Promote error:', err);
      res.status(500).json({ error: 'Server error promoting user' });
    }
  });

  app.delete('/api/reports/:id', protect, authorize('admin'), async (req, res) => {
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

  // Admin-only Routes
  app.get('/api/admin/users', protect, authorize('admin'), async (req, res) => {
    try {
      const users = await User.find().select('-password');
      res.status(200).json({ success: true, data: users });
    } catch (err) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  // Utility Routes
  app.get('/api/trains', protect, async (req, res) => {
    try {
      const trains = await Report.distinct('trainNumber');
      res.json(trains);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get('/api/compartments/:trainNumber', protect, async (req, res) => {
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
    
    if (err instanceof mongoose.Error.ValidationError) {
      const messages = Object.values(err.errors).map(val => val.message);
      return res.status(400).json({
        success: false,
        error: 'Validation error',
        messages: messages
      });
    }
    
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
