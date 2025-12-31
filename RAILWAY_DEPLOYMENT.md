# 🚂 Railway Deployment Guide

## ✅ Prerequisites

- [x] GitHub account
- [x] Railway account (sign up at [railway.app](https://railway.app))
- [x] Your Google API keys ready
- [x] Project pushed to GitHub

---

## 🚀 Quick Deployment (5 Minutes)

### Step 1: Prepare Your Repository

```bash
# Make sure you're in the project directory
cd ~/Documents/4rth\ Year/NLP/Project/Research-Production-Analysis-System

# Check your current branch
git branch
# Note which branch you're on (e.g., dev, develop, your-branch-name)

# Check if files exist
ls Dockerfile railway.toml
# You should see both files ✅

# Add and commit to YOUR branch
git add Dockerfile railway.toml
git commit -m "Add Railway deployment configuration"
git push origin <your-branch-name>
# Replace <your-branch-name> with your actual branch (e.g., git push origin dev)
```

### Step 2: Create Railway Project

1. **Go to [railway.app](https://railway.app)**
2. **Click "Login"** → Sign in with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**: `Research-Production-Analysis-System`
6. **⚠️ IMPORTANT: Select your branch**
   - Railway defaults to `main` branch
   - If your code is in a different branch (e.g., `dev`, `develop`, `feature-branch`):
     - Click "Configure" or "Settings" after selecting repo
     - Under "Source", select your branch from dropdown
     - Or skip this step and configure after deployment (see below)
7. **Click "Deploy Now"**

#### If You Need to Change Branch After Deployment:

1. **Go to your service** in Railway dashboard
2. **Click "Settings" tab**
3. **Scroll to "Source"** section
4. **Change "Branch"** dropdown to your branch name
5. **Click "Redeploy"**

### Step 3: Configure Environment Variables

In the Railway dashboard:

1. **Click on your service** (should say "Deploying...")
2. **Go to "Variables" tab**
3. **Click "Add Variable"**
4. **Add your API keys:**

```
Variable Name: GOOGLE_API_KEYS
Value: your-key-1,your-key-2,your-key-3,your-key-4
```

5. **Click "Add"**

### Step 4: Wait for Deployment

- Railway will automatically:
  - ✅ Build your Docker image
  - ✅ Install dependencies
  - ✅ Start the services
  - ✅ Generate a public URL

- **Build time**: 3-5 minutes
- **Watch logs**: Click "Deployments" tab → Click on the deployment → View logs

### Step 5: Get Your URL

1. **Go to "Settings" tab**
2. **Scroll to "Domains"**
3. **Click "Generate Domain"**
4. **Copy your URL**: `https://your-project-name.up.railway.app`

### Step 6: Test Your API

```bash
# Test health endpoint
curl https://your-project-name.up.railway.app/health

# Expected response:
# {"status": "healthy"}
```

```bash
# Test classification
curl -X POST https://your-project-name.up.railway.app/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Image Recognition",
    "abstract": "This paper presents a convolutional neural network for image classification tasks..."
  }'
```

---

## 🎯 You're Done! 

Your RAG system is now live at:
```
https://your-project-name.up.railway.app
```

---

## 📊 Railway Dashboard Overview

### Important Tabs:

1. **Deployments** - View build logs, deployment history
2. **Metrics** - CPU, memory, network usage
3. **Variables** - Environment variables (API keys)
4. **Settings** - Domain, restart service, delete project
5. **Observability** - Logs in real-time

---

## 💰 Free Tier Details

### What You Get:
- ✅ **$5 credit/month** (≈500 execution hours)
- ✅ **512MB RAM**
- ✅ **1 vCPU**
- ✅ **100GB bandwidth**
- ✅ **Persistent storage**

### Usage Monitoring:
- Check usage: Dashboard → Click on project → "Usage" tab
- You'll see: CPU hours, Network, Memory
- Free tier resets every month

### Tips to Save Credits:
- Service auto-sleeps when inactive (saves money)
- Only runs when receiving requests
- Monitor usage regularly

---

## 🔧 Configuration Files Explained

### `Dockerfile`
- Builds Python 3.11 environment
- Installs system dependencies (curl, docker)
- Copies your project files
- Installs Python packages from requirements.txt
- Exposes port 5000
- Runs health checks every 30s
- Starts Flask backend

### `railway.toml`
- Tells Railway to use Docker
- Sets health check endpoint: `/health`
- Configures restart policy (auto-restart on failure)
- Sets startup command

---

## 🐛 Troubleshooting

### "Build Failed"

**Check logs:**
```
Railway Dashboard → Deployments → Click failed deployment → View logs
```

**Common issues:**
- Missing dependencies in `requirements.txt`
- Syntax error in Dockerfile
- Out of memory during build

**Solution:**
```bash
# Test Docker build locally first
docker build -t rag-test .
docker run -p 5000:5000 rag-test
```

### "Service Crashed"

**Check runtime logs:**
```
Railway Dashboard → Observability → Logs
```

**Common causes:**
- Missing `GOOGLE_API_KEYS` environment variable
- API key quota exceeded
- Out of memory (512MB limit)

**Solution:**
- Verify environment variables are set
- Check API key validity
- Consider upgrading plan if needed

### "502 Bad Gateway"

**Cause:** Service not ready yet

**Solution:**
- Wait 30-60 seconds after deployment
- Check if health check passes: `/health` endpoint
- View logs for startup errors

### "Out of Credits"

**Check usage:**
```
Dashboard → Project → Usage tab
```

**Solutions:**
- Wait for monthly reset (1st of each month)
- Upgrade to paid plan ($5/month for 500 hours + $0.01/hour after)
- Optimize service (reduce idle time)

---

## 📈 Monitoring Your Service

### View Real-Time Logs:
```
Railway Dashboard → Observability → Logs
```

You'll see:
- Flask startup messages
- GROBID initialization
- ChromaDB loading
- API requests
- Errors

### Check Metrics:
```
Railway Dashboard → Metrics
```

Monitor:
- CPU usage
- Memory usage
- Network traffic
- Request count

### Set Up Alerts:
- Dashboard → Settings → Notifications
- Get notified on:
  - Deployment failures
  - Service crashes
  - High resource usage

---

## 🔄 Updating Your Deployment

### Automatic Deployments:

Railway auto-deploys when you push to GitHub:

```bash
# Make changes to your code
nano backend/app.py

# Commit and push
git add .
git commit -m "Update backend logic"
git push origin main

# Railway will automatically:
# 1. Detect push
# 2. Rebuild Docker image
# 3. Deploy new version
# 4. Zero-downtime switch
```

### Manual Deployments:

```
Railway Dashboard → Deployments → "Deploy" button
```

---

## 🌐 Custom Domain (Optional)

### Add Your Own Domain:

1. **Go to Settings → Domains**
2. **Click "Add Custom Domain"**
3. **Enter your domain**: `api.yourdomain.com`
4. **Add CNAME record** to your DNS:
   ```
   CNAME: api.yourdomain.com → your-project.up.railway.app
   ```
5. **Wait for DNS propagation** (5-30 minutes)
6. **Railway auto-generates SSL certificate** ✅

Your API will be available at:
```
https://api.yourdomain.com
```

---

## 🔒 Security Best Practices

### Environment Variables:
- ✅ Never commit API keys to Git
- ✅ Use Railway's "Variables" feature
- ✅ Keep `.env` in `.gitignore`

### API Security:
```python
# Add to backend/app.py if needed
from flask import request, abort

@app.before_request
def check_auth():
    api_key = request.headers.get('X-API-Key')
    if api_key != os.getenv('API_SECRET'):
        abort(401)
```

### CORS Configuration:
```python
# In backend/app.py
from flask_cors import CORS

# Only allow your frontend domain
CORS(app, origins=[
    "https://your-frontend.com",
    "http://localhost:3000"  # for development
])
```

---

## 📊 Performance Optimization

### Reduce Cold Starts:
```
Railway Dashboard → Settings → Always On (Paid feature)
```

### Optimize Docker Build:
```dockerfile
# Use build cache efficiently
# Copy requirements first (changes less often)
COPY backend/requirements.txt ./backend/
RUN pip install -r backend/requirements.txt

# Then copy code (changes more often)
COPY backend/ ./backend/
```

### Monitor Response Times:
```python
# Add to backend/app.py
import time

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    print(f"Request took {duration:.2f}s")
    return response
```

---

## 💡 Tips & Tricks

### 1. **View Database Contents:**
```bash
# SSH into Railway service (Hobby plan+)
railway run bash

# Or check locally before deploying
python3 -c "from RAG.vector_db_manager import VectorDBManager; db = VectorDBManager(); print(db.collection.count())"
```

### 2. **Test Before Deploying:**
```bash
# Build and run Docker locally
docker build -t rag-local .
docker run -p 5000:5000 --env-file RAG/.env rag-local

# Test endpoint
curl http://localhost:5000/health
```

### 3. **Quick Rollback:**
```
Railway Dashboard → Deployments → Click previous deployment → "Redeploy"
```

### 4. **Scale Up (Paid Plans):**
- More RAM (up to 32GB)
- More vCPUs
- Multiple regions
- Always-on (no cold starts)

---

## 📞 Quick Commands Reference

### Deploy from CLI (Optional):

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# Deploy
railway up

# View logs
railway logs

# Open in browser
railway open
```

---

## ✅ Deployment Checklist

- [ ] GitHub repo is up-to-date
- [ ] `Dockerfile` exists in root
- [ ] `railway.toml` exists in root
- [ ] Pushed to GitHub
- [ ] Created Railway project
- [ ] Connected GitHub repo
- [ ] Added `GOOGLE_API_KEYS` environment variable
- [ ] Deployment succeeded
- [ ] Generated domain
- [ ] Tested `/health` endpoint
- [ ] Tested `/api/classify` endpoint
- [ ] Updated frontend to use Railway URL
- [ ] Monitored first few requests
- [ ] Checked credits usage

---

## 🎉 Success!

Your RAG classification system is now live on Railway!

### Your API Endpoints:
```
Base URL: https://your-project-name.up.railway.app

GET  /health                    - Health check
POST /api/classify              - Classify article
POST /api/upload                - Upload PDF
POST /api/process               - Complete pipeline
```

### Next Steps:
1. ✅ Test all endpoints
2. ✅ Connect your frontend
3. ✅ Monitor usage in Railway dashboard
4. ✅ Set up error alerts
5. ✅ Share your API with team/professors

---

## 📚 Additional Resources

- **Railway Docs**: https://docs.railway.app
- **Railway Community**: https://discord.gg/railway
- **Status Page**: https://status.railway.app
- **Pricing**: https://railway.app/pricing

---

## 🆘 Need Help?

### Check logs first:
```
Railway Dashboard → Observability → Logs
```

### Common fixes:
- Restart service: Settings → "Restart"
- Redeploy: Deployments → "Deploy"
- Clear build cache: Settings → "Clear Build Cache"

### Get support:
- Railway Discord: https://discord.gg/railway
- GitHub Issues: Your repository issues
- Railway Support: help@railway.app

---

**Happy Deploying! 🚂🚀**

Your system will be live in ~5 minutes!
