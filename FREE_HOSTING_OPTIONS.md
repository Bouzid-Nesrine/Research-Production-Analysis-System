# 🆓 Free Hosting Options for Your RAG System

## 📊 Quick Comparison

| Platform | Free Tier | Docker Support | Best For | Limitations |
|----------|-----------|----------------|----------|-------------|
| **Render** ✅ | Yes, permanent | ✅ Yes | Flask backend | 750 hours/month, sleeps after inactivity |
| **Railway** ✅ | $5 credit/month | ✅ Yes | Full stack | 500 hours/month, credit-based |
| **Fly.io** ✅ | Yes, permanent | ✅ Yes | Docker apps | 3 shared VMs, 160GB bandwidth |
| **Oracle Cloud** ⭐ | Always free | ✅ Yes | Complete system | Best specs, requires credit card |
| **Google Cloud** | $300 credit (90 days) | ✅ Yes | Everything | Requires credit card, then paid |
| **PythonAnywhere** | Yes, limited | ❌ No | Simple Flask | No Docker, 512MB RAM |
| **Replit** | Yes, limited | ❌ No | Development | Not for production |

---

## ⭐ RECOMMENDED: Best Free Options

### 🥇 Option 1: Oracle Cloud (BEST - Always Free)

**Why Oracle Cloud:**
- ✅ **Always Free** (not a trial)
- ✅ 2 AMD VMs with 1GB RAM each OR 4 Arm VMs with 24GB RAM total
- ✅ Full root access
- ✅ Run Docker, Flask, everything
- ✅ 10TB outbound bandwidth/month
- ✅ No automatic charges

**Requirements:**
- Credit card (for verification only, never charged)
- Oracle account

**Setup Steps:**

```bash
# 1. Create Oracle Cloud account
# Visit: https://www.oracle.com/cloud/free/

# 2. Create VM Instance
# - Choose: VM.Standard.E2.1.Micro (Always Free)
# - OS: Ubuntu 22.04
# - Note the public IP

# 3. SSH to your instance
ssh ubuntu@<your-public-ip>

# 4. Install Docker
sudo apt update
sudo apt install -y docker.io python3-pip git
sudo usermod -aG docker ubuntu
exit
# SSH again to apply docker permissions

# 5. Clone your project
git clone <your-repo-url>
cd Research-Production-Analysis-System

# 6. Setup environment
nano RAG/.env
# Add: GOOGLE_API_KEYS=key1,key2,key3,key4

# 7. Start system
./start_system.sh

# 8. Open firewall ports
# In Oracle Cloud Console:
# Networking → Virtual Cloud Networks → Security Lists
# Add Ingress Rules:
# - Port 5000 (Flask)
# - Port 8070 (GROBID)
```

**Access your API:**
```
http://<oracle-vm-public-ip>:5000
```

**Cost:** $0/month forever ✅

---

### 🥈 Option 2: Render (Easiest Setup)

**Why Render:**
- ✅ Easiest deployment (Git-based)
- ✅ Free Docker support
- ✅ Automatic HTTPS
- ✅ No credit card needed

**Limitations:**
- ⚠️ Sleeps after 15 mins inactivity (30s cold start)
- ⚠️ 750 hours/month free tier
- ⚠️ 512MB RAM limit

**Setup Steps:**

```bash
# 1. Create render.yaml in project root
cat > render.yaml << 'EOF'
services:
  - type: web
    name: rag-backend
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: GOOGLE_API_KEYS
        sync: false
    healthCheckPath: /health
    plan: free
EOF

# 2. Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r RAG/requirements.txt

# Expose port
EXPOSE 5000

# Start script
CMD ["./start_system.sh"]
EOF

# 3. Push to GitHub
git add .
git commit -m "Add Render deployment"
git push origin main

# 4. Connect Render
# Visit: https://render.com
# - Sign up (free)
# - New → Web Service → Connect GitHub repo
# - Select your repository
# - Add environment variable: GOOGLE_API_KEYS

# 5. Deploy automatically
```

**Your API will be at:**
```
https://rag-backend.onrender.com
```

**Cost:** $0/month ✅

---

### 🥉 Option 3: Railway (Good Free Tier)

**Why Railway:**
- ✅ $5 free credit/month (≈500 hours)
- ✅ Great Docker support
- ✅ Simple deployment
- ✅ No sleep mode

**Limitations:**
- ⚠️ Credit-based (runs out if heavy usage)
- ⚠️ Requires GitHub account

**Setup Steps:**

```bash
# 1. Create railway.json
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "./start_system.sh",
    "healthcheckPath": "/health"
  }
}
EOF

# 2. Push to GitHub
git add railway.json
git commit -m "Add Railway deployment"
git push

# 3. Deploy
# Visit: https://railway.app
# - Login with GitHub
# - New Project → Deploy from GitHub
# - Select repository
# - Add environment variable: GOOGLE_API_KEYS
# - Deploy
```

**Your API will be at:**
```
https://<your-project>.up.railway.app
```

**Cost:** $0/month (with $5 credit) ✅

---

### 🥉 Option 4: Fly.io (Developer Friendly)

**Why Fly.io:**
- ✅ Free tier: 3 shared-CPU VMs
- ✅ Excellent Docker support
- ✅ Global deployment
- ✅ No sleep mode

**Limitations:**
- ⚠️ 160GB outbound bandwidth/month
- ⚠️ Requires credit card

**Setup Steps:**

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
flyctl auth login

# 3. Create fly.toml
cat > fly.toml << 'EOF'
app = "rag-classification"

[env]
  PORT = "5000"

[http_service]
  internal_port = 5000
  force_https = true
  auto_start_machines = true
  auto_stop_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
EOF

# 4. Launch
flyctl launch

# 5. Set secrets
flyctl secrets set GOOGLE_API_KEYS="key1,key2,key3,key4"

# 6. Deploy
flyctl deploy
```

**Your API will be at:**
```
https://rag-classification.fly.dev
```

**Cost:** $0/month ✅

---

## 💡 Budget Options Comparison

### For Your RAG System Specifically:

| Component | Oracle Cloud | Render | Railway | Fly.io |
|-----------|-------------|---------|---------|---------|
| **GROBID Docker** | ✅ Works | ✅ Works | ✅ Works | ✅ Works |
| **Flask Backend** | ✅ Works | ✅ Works | ✅ Works | ✅ Works |
| **ChromaDB** | ✅ Persistent | ⚠️ Limited | ⚠️ Limited | ✅ Volumes |
| **RAM** | 1GB-24GB | 512MB | 512MB-1GB | 256MB-1GB |
| **Uptime** | 24/7 | Sleeps | 24/7 (credit) | 24/7 |
| **Setup Difficulty** | Medium | Easy | Easy | Medium |

---

## 🎯 My Recommendations

### If You Want ZERO Hassle:
**Use Render** - Push code, automatic deployment, free HTTPS
```bash
# Just push to GitHub and connect Render
git push origin main
# Done! ✅
```

### If You Want Best Performance:
**Use Oracle Cloud** - Full VM, no limitations, always free
```bash
# One-time SSH setup, then runs forever
ssh ubuntu@<ip>
./start_system.sh
# Runs 24/7 forever ✅
```

### If You Want Balance:
**Use Railway** - Good free tier, no sleep mode, easy setup
```bash
# Git-based deployment with monitoring
flyctl deploy
# Runs until credit depletes ✅
```

---

## 🚀 Fastest Way to Get Started (Under 5 Minutes)

### **Render Quickstart:**

```bash
# 1. Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y docker.io
COPY . .
RUN pip install -r backend/requirements.txt
RUN pip install -r RAG/requirements.txt
EXPOSE 5000
CMD ["python", "backend/app.py"]
EOF

# 2. Push to GitHub
git add Dockerfile
git commit -m "Add Dockerfile"
git push

# 3. Go to render.com
# - Sign up
# - New Web Service
# - Connect GitHub
# - Select repo
# - Add env var: GOOGLE_API_KEYS
# - Deploy

# DONE! 🎉
```

**Live in 3-5 minutes!**

---

## 📋 Free Hosting Checklist

- [ ] Choose platform (I recommend **Render** for easiest or **Oracle** for best)
- [ ] Create account (no credit card for Render/Railway)
- [ ] Push code to GitHub (if using Render/Railway)
- [ ] Add API keys as environment variables
- [ ] Deploy (one command or web UI)
- [ ] Test endpoint: `https://your-app-url.com/health`
- [ ] Update frontend to use new API URL

---

## ⚠️ Important Notes

### Database Persistence:
- **Oracle Cloud**: Full persistence ✅
- **Render**: Limited (files may be lost on restart) ⚠️
- **Railway**: Volume storage available ✅
- **Fly.io**: Volumes available ✅

**Solution for Render/Railway:**
- ChromaDB rebuilds automatically from taxonomy files
- Takes ~30 seconds on first request after sleep
- Or use external storage (S3, etc.)

### API Key Security:
- ✅ Use environment variables (never commit to Git)
- ✅ All platforms support secret environment variables
- ✅ Add `.env` to `.gitignore`

### GROBID Docker:
- Oracle Cloud: Full Docker support ✅
- Render: Supported ✅
- Railway: Supported ✅
- Fly.io: Supported ✅

---

## 🎓 Step-by-Step: Render (Recommended for Students)

```bash
# Step 1: Prepare your project
cd ~/Documents/4rth\ Year/NLP/Project/Research-Production-Analysis-System

# Step 2: Create Dockerfile (if not exists)
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY backend/ ./backend/
COPY RAG/ ./RAG/
COPY start_system.sh ./

# Install Python packages
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r RAG/requirements.txt

EXPOSE 5000

# Run
CMD ["python", "backend/app.py"]
EOF

# Step 3: Commit
git add Dockerfile
git commit -m "Add Dockerfile for deployment"
git push origin main

# Step 4: Deploy on Render
# 1. Go to https://render.com
# 2. Sign up (free, use GitHub)
# 3. Dashboard → New → Web Service
# 4. Connect your GitHub repo
# 5. Settings:
#    - Name: rag-backend
#    - Environment: Docker
#    - Plan: Free
# 6. Environment Variables:
#    - Key: GOOGLE_API_KEYS
#    - Value: your,api,keys,here
# 7. Click "Create Web Service"

# Step 5: Wait 3-5 minutes for deployment

# Step 6: Test
curl https://rag-backend.onrender.com/health
```

**Your free API is live!** 🎉

---

## 💰 Cost Summary

| Platform | Monthly Cost | Best Use Case |
|----------|-------------|---------------|
| Oracle Cloud | **$0** | Production, 24/7 uptime |
| Render | **$0** | Demo, portfolio, light usage |
| Railway | **$0** | Development, testing |
| Fly.io | **$0** | Small production apps |

**All options are FREE!** 🎉

---

## 🆘 Need Help?

### Common Issues:

**"Out of memory"**
- Reduce ChromaDB size
- Use smaller embedding model
- Upgrade to paid tier

**"Service sleeps"**
- Use Oracle Cloud or Railway
- Or accept 30s cold start on Render

**"Docker not supported"**
- Use Oracle Cloud or Fly.io
- Or deploy without Docker

---

## 🎯 Final Recommendation

**For Your NLP Project:**

1. **Development/Demo**: Use **Render**
   - Easiest setup
   - Free HTTPS
   - Good for portfolio

2. **Production/Research**: Use **Oracle Cloud**
   - Always free
   - Best performance
   - 24/7 uptime

3. **Quick Testing**: Use **Railway**
   - Fast deployment
   - Good monitoring

**Start with Render today, migrate to Oracle Cloud if needed later.**

---

## 📞 Quick Deploy Commands

### Render (Easiest):
```bash
# Just push to GitHub, then connect on render.com
git push origin main
```

### Oracle Cloud (Best Free Tier):
```bash
ssh ubuntu@<oracle-ip>
git clone <your-repo>
cd Research-Production-Analysis-System
./start_system.sh
```

### Railway:
```bash
railway login
railway init
railway up
```

### Fly.io:
```bash
flyctl launch
flyctl deploy
```

---

**Choose Render → Deploy in 5 minutes → Get your free API! 🚀**
