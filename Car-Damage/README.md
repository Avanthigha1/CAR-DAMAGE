# Car Damage Detection & Real-Time Valuation System

## Project Overview

A full-stack web application that leverages computer vision to detect vehicle damage from images and provides real-time market valuation with damage adjustments. The system analyzes car damage severity and integrates with market data to estimate accurate vehicle values.

### Key Features

- **AI-Powered Damage Detection**: Uses OpenCV for edge detection and contour analysis
- **Real-Time Valuation**: Fetches current market prices and calculates damage adjustments
- **Severity Classification**: Categorizes damage as minor, moderate, or severe
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Production-Ready**: Docker support, comprehensive logging, error handling
- **RESTful APIs**: Clean, documented endpoints for damage detection and valuation

---

## Tech Stack

### Backend

- **Framework**: Flask 2.3.3
- **Computer Vision**: OpenCV 4.8.0
- **Image Processing**: NumPy 1.24.3
- **Web Scraping**: BeautifulSoup4, Requests
- **Server**: Gunicorn 21.2.0
- **CORS**: Flask-CORS 4.0.0

### Frontend

- **HTML5**, **CSS3**, **Vanilla JavaScript**
- **Responsive Design** with CSS Grid
- **Real-time UI Updates** with Fetch API
- **Modern UX** with animations and loading states

### DevOps & Deployment

- **Containerization**: Docker 24+
- **Orchestration**: Docker Compose
- **Cloud Platforms**: Heroku, Railway, Vercel
- **CI/CD**: GitHub Actions
- **Version Control**: Git

### Database (Optional)

- PostgreSQL for future enhancement
- Redis for caching

---

## Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)
- 2GB RAM minimum

---

## Installation & Setup

### Local Development Setup

#### 2. Create Virtual Environment

python -m venv venv

# On Windows

venv\Scripts\activate

# On macOS/Linux

source venv/bin/activate

````

3. Install Dependencies

```bash
pip install -r requirements.txt
````

#### 4. Environment Configuration

Create a `.env` file in the project root:

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=16777216
```

#### 5. Create Uploads Directory

```bash
mkdir uploads
```

#### 6. Run Application

```bash
python app.py
```

Application will be available at `http://localhost:5000`

---

## Docker Deployment

### Local Docker Build

```bash
# Build image
docker build -t car-damage-app:latest .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -e FLASK_ENV=production \
  car-damage-app:latest
```

### Docker Compose

```bash
docker-compose up -d
```

Access at `http://localhost:5000`

---

## Cloud Deployment

### Heroku Deployment

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-production-secret-key

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Railway Deployment

1. Connect GitHub repository to Railway
2. Add environment variables in Railway dashboard
3. Railway automatically deploys on push to main branch

### Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Configure serverless functions if needed
```

---

## 📚 API Documentation

### 1. Damage Detection Endpoint

**POST** `/api/detect-damage`

Upload an image and receive damage analysis.

**Request**:

```
Content-Type: multipart/form-data
- image: [binary image file]
```

**Response** (200 OK):

```json
{
  "status": "success",
  "damage_severity": "moderate",
  "num_damaged_regions": 3,
  "annotated_image": "/uploads/annotated_20240115_143022_car.jpg",
  "multiplier": 0.7
}
```

**Error Response** (400/500):

```json
{
  "error": "Invalid file format"
}
```

---

### 2. Value Estimation Endpoint

**POST** `/api/estimate-value`

Calculate vehicle value based on damage and market data.

**Request**:

```json
{
  "make": "Maruti",
  "model": "Swift",
  "year": 2020,
  "mileage": 45000,
  "damage_severity": "moderate"
}
```

**Response** (200 OK):

```json
{
  "status": "success",
  "market_value": 650000,
  "damage_multiplier": 0.7,
  "final_value": 455000,
  "deduction": 195000,
  "similar_cars": [
    {
      "year": 2020,
      "make": "Maruti",
      "model": "Swift",
      "price": 650000
    }
  ]
}
```

---

### 3. Health Check Endpoint

**GET** `/api/health`

Verify application is running.

**Response** (200 OK):

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:22.123456"
}
```

---

## Damage Detection Algorithm

### Process Flow

1. **Image Upload**: Validate file format and size
2. **Preprocessing**:
   - Convert to grayscale
   - Apply Gaussian blur (5x5 kernel)
3. **Edge Detection**: Canny edge detection (50-150 threshold)
4. **Morphological Operations**: Close operation to enhance regions
5. **Contour Detection**: Find and filter contours by area
6. **Severity Classification**:
   - 0 regions: No damage
   - 1-2 regions: Minor (0.9x multiplier)
   - 3-5 regions: Moderate (0.7x multiplier)
   - 6+ regions: Severe (0.5x multiplier)
7. **Visualization**: Draw bounding boxes on detected damage

### Key Parameters

- **Min Contour Area**: 100 pixels²
- **Max Contour Area**: 30% of image area
- **Canny Thresholds**: 50 (lower), 150 (upper)
- **Morphological Kernel**: 5x5 rectangle

---

## 💰 Valuation Algorithm

### Market Value Calculation

1. **Age-Based Depreciation**:

   - 0-2 years: 85% retention
   - 2-5 years: 70% retention
   - 5-10 years: 50% retention
   - 10+ years: 30% retention

2. **Damage Adjustment**:
   - Final Value = Market Value × Damage Multiplier

### Example

```
Base Price (Sedan): ₹250,000
Year: 2020 (4 years old)
Depreciation: 70% → Market Value: ₹175,000
Damage Severity: Moderate (0.7x)
Final Value: ₹175,000 × 0.7 = ₹122,500
Deduction: ₹52,500
```

---

## 📁 Project Structure

```
car-damage-detection/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── Procfile                    # Heroku deployment config
├── runtime.txt                 # Python version for Heroku
├── railway.toml               # Railway deployment config
├── .env                       # Environment variables (not in git)
├── .gitignore                 # Git ignore rules
│
├── templates/
│   └── index.html             # Frontend HTML
│
├── uploads/                   # Uploaded images directory
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Utility functions
│   ├── damage_detector.py     # CV algorithm
│   └── valuation.py           # Pricing logic
│
├── tests/
│   ├── test_damage_detection.py
│   ├── test_valuation.py
│   └── test_api.py
│
├── docs/
│   ├── API.md                 # API documentation
│   ├── ARCHITECTURE.md        # System architecture
│   └── DEPLOYMENT.md          # Deployment guide
│
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD pipeline
│
└── README.md                  # This file
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/test_damage_detection.py -v
```

### Integration Tests

```bash
pytest tests/test_api.py -v
```

### Coverage Report

```bash
pytest --cov=src tests/
```

---

## 📊 Performance Optimization

### Image Processing

- Resize large images before processing
- Use efficient contour detection algorithms
- Cache market data for common queries

### Caching Strategy

- Cache market values for 24 hours
- Use Redis for distributed caching
- Implement request deduplication

### API Optimization

- Implement pagination for similar cars
- Use connection pooling for database queries
- Compress API responses with gzip

---

## 🔒 Security Features

### Input Validation

- File type and size validation
- File extension verification
- MIME type checking

### Error Handling

- No sensitive data in error messages
- Proper exception logging
- Graceful error responses

### CORS Configuration

- Restricted to trusted domains
- Configurable in production

### Environment Security

- Secrets stored in `.env` (never committed)
- Different keys for dev/prod
- Secure header configuration

---

## 📈 Monitoring & Logging

### Application Logging

- All API requests logged
- Error stack traces captured
- Performance metrics tracked

### Log Files

```bash
# View logs
tail -f app.log

# Rotate logs (via logrotate)
logrotate -f /etc/logrotate.d/app
```

### Metrics to Monitor

- API response times
- Error rates
- Image processing time
- Database query performance

---

## 🐛 Known Limitations

1. **Damage Detection**:

   - Works best with clear, well-lit images
   - May have false positives with reflections/shadows
   - Limited to visible external damage

2. **Valuation**:

   - Based on simplified depreciation model
   - Regional price variations not fully captured
   - Market data refresh rate: manual updates

3. **Deployment**:
   - OpenCV requires system libraries (libsm6, libxext6)
   - File uploads limited to 16MB per image
   - Concurrent upload limit based on available memory

---

## 🚀 Future Enhancements

### Phase 2

- [ ] Machine learning model (ResNet) for damage classification
- [ ] Integration with real car marketplace APIs (OLX, Cars24)
- [ ] Advanced damage severity scoring using ML
- [ ] Historical pricing data and trends

### Phase 3

- [ ] User authentication and dashboard
- [ ] Valuation report generation (PDF export)
- [ ] Vehicle comparison tool
- [ ] Insurance claim integration

### Phase 4

- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time price notifications
- [ ] AI-powered damage estimation with photos
- [ ] Integration with repair shops for cost estimates

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: OpenCV library not found

```bash
pip install opencv-python
```

**Issue**: Port 5000 already in use

```bash
# Change port in app.py or use environment variable
export FLASK_PORT=5001
```

**Issue**: Image upload fails

- Verify file size < 16MB
- Check file format (JPG, PNG, GIF)
- Ensure uploads directory has write permissions

**Issue**: Docker build fails

```bash
# Clear Docker cache
docker system prune -a
docker build --no-cache -t car-damage-app:latest .
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style

- Follow PEP 8 for Python
- Use meaningful variable names
- Add docstrings to functions
- Keep functions under 50 lines

---

## 📧 Contact

For questions, suggestions, or issues, please:

- Open an issue on GitHub
- Contact: your-email@example.com

---

## 🙏 Acknowledgments

- OpenCV community for computer vision library
- Flask community for web framework
- Contributors and testers
