# Railway Deployment Guide

## 🚀 Quick Deploy to Railway

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
```

### Step 2: Initialize Git Repository
```bash
cd /home/desal/Archangel
git init
git add .
git commit -m "Initial commit: Archangel AI Monetization System"
```

### Step 3: Deploy to Railway
```bash
# Login to Railway
railway login

# Initialize Railway project
railway init

# Deploy
railway up
```

### Step 4: Set Environment Variables
In Railway dashboard, add these environment variables:

**Required:**
```
JWT_SECRET=your-super-secret-jwt-key-here
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

**Optional (for full functionality):**
```
STRIPE_SECRET_KEY=sk_test_your-stripe-key
STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-key
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Step 5: Access Your API
- Your app will be available at: `https://yourapp.railway.app`
- API docs: `https://yourapp.railway.app/docs`
- Health check: `https://yourapp.railway.app/health`

## 🔧 Configuration

### Environment Variables in Railway:
1. Go to your Railway dashboard
2. Select your project
3. Click "Variables" tab
4. Add the environment variables listed above

### Database:
- Railway will automatically provision a PostgreSQL database
- The `DATABASE_URL` will be set automatically
- For development, SQLite is used (no setup required)

## 📚 Testing Your Deployment

### 1. Health Check
```bash
curl https://yourapp.railway.app/health
```

### 2. List Models
```bash
curl https://yourapp.railway.app/v1/models
```

### 3. Create User (via API)
```bash
curl -X POST https://yourapp.railway.app/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'
```

### 4. Test Chat Completion
```bash
curl -X POST https://yourapp.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 🔒 Production Checklist

### Security:
- [ ] Set strong JWT_SECRET
- [ ] Enable HTTPS (automatic on Railway)
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Add API key authentication

### Monitoring:
- [ ] Set up error tracking (Sentry)
- [ ] Configure logging
- [ ] Monitor usage and costs
- [ ] Set up alerts

### Business:
- [ ] Configure Stripe for payments
- [ ] Set up webhook endpoints
- [ ] Configure email notifications
- [ ] Add terms of service

## 🛠️ Common Issues

### 1. Database Connection
If you get database errors, Railway may still be provisioning PostgreSQL. Wait a few minutes or check the Railway dashboard.

### 2. Missing Environment Variables
Make sure all required environment variables are set in Railway dashboard.

### 3. Build Failures
Check the Railway build logs for specific error messages.

### 4. Port Issues
Railway automatically sets the PORT environment variable. The app is configured to use it.

## 📈 Scaling

Railway automatically handles:
- Load balancing
- Auto-scaling
- SSL certificates
- CDN
- Database backups

For high-traffic scenarios, consider:
- Redis for caching
- Separate database instances
- Multiple regions
- Rate limiting optimization

## 💰 Costs

Railway pricing:
- **Hobby Plan**: $5/month (includes databases)
- **Pro Plan**: $20/month (more resources)
- **Usage-based**: Additional charges for high resource usage

Typical monthly costs for AI API:
- Small scale: $5-20/month
- Medium scale: $20-100/month
- Large scale: $100+/month