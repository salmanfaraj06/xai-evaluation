# 🚀 HEXEval Deployment Guide

Complete guide for deploying HEXEval to **Streamlit Community Cloud** with CI/CD pipeline.

---

## 📋 Prerequisites

- [x] GitHub account
- [x] Git installed locally
- [x] OpenAI API key (for LLM persona evaluation)
- [x] Python 3.9+ installed locally (for testing)

---

## 🔐 Step 1: Security Setup (CRITICAL)

### Remove Exposed Secrets

The `.env` file has been sanitized, but you need to add your actual API key for local development:

1. **Open `.env` file**
2. **Replace placeholder** with your actual OpenAI API key:
   ```bash
   OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY-HERE
   ```

3. **Verify `.env` is in `.gitignore`**:
   ```bash
   cat .gitignore | grep .env
   ```
   Should show: `.env`

> [!CAUTION]
> **NEVER commit your actual API key to Git!** The `.env` file should remain in `.gitignore`.

---

## 🧪 Step 2: Test Locally

Before deploying, verify everything works locally:

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=hexeval
```

### Test Streamlit App

```bash
# Run the app locally
streamlit run hexeval/ui/app.py
```

**Verify:**
- ✅ App loads without errors
- ✅ Both demo use cases work (Healthcare & Credit Risk)
- ✅ All tabs render correctly
- ✅ Configuration options work

---

## 📤 Step 3: Push to GitHub

### Commit Changes

```bash
# Check status
git status

# Add all deployment files
git add .
git commit -m "Add Streamlit Cloud deployment configuration with CI/CD"

# Push to GitHub
git push origin hexeval  # or your branch name
```

### Verify GitHub Actions

1. Go to your repository on GitHub
2. Click **"Actions"** tab
3. Verify workflows are running:
   - ✅ **Test Suite** - Should pass
   - ✅ **Code Quality** - May show warnings (optional)

> [!NOTE]
> If tests fail, check the logs and fix issues before deploying.

---

## ☁️ Step 4: Deploy to Streamlit Cloud

### 4.1 Sign Up / Log In

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories

### 4.2 Create New App

1. Click **"New app"** button
2. Fill in the form:
   - **Repository**: `salmanfaraj06/xai-evaluation`
   - **Branch**: `hexeval` (or `main`)
   - **Main file path**: `hexeval/ui/app.py`
   - **App URL**: Choose a custom URL (e.g., `hexeval-demo`)

3. Click **"Advanced settings"** (optional):
   - **Python version**: `3.10` (matches runtime.txt)
   - Leave other settings as default

### 4.3 Configure Secrets

> [!IMPORTANT]
> This is **required** for LLM persona evaluation to work!

1. In the app settings, click **"Secrets"** in the left sidebar
2. Add your OpenAI API key in TOML format:

```toml
OPENAI_API_KEY = "sk-proj-YOUR-ACTUAL-KEY-HERE"
```

3. Click **"Save"**

### 4.4 Deploy

1. Click **"Deploy!"**
2. Wait for deployment (usually 2-5 minutes)
3. Watch the logs for any errors

---

## ✅ Step 5: Verify Deployment

### Test the Live App

Once deployed, test all functionality:

#### Healthcare Use Case
1. Select **"Heart Disease (Healthcare)"**
2. Enable **"Use sample model and dataset"**
3. Click **"🚀 Run Evaluation"**
4. Verify results appear in **Results** and **Recommendations** tabs

#### Credit Risk Use Case
1. Select **"Credit Risk (Finance)"**
2. Enable **"Use sample model and dataset"**
3. Click **"🚀 Run Evaluation"**
4. Verify results appear

#### Test Without API Key
1. Disable **"Enable LLM personas"** in sidebar
2. Run evaluation
3. Should work but skip persona evaluation

#### Test All Tabs
- ✅ Configuration & Run
- ✅ Use Case Details (shows personas)
- ✅ Results (technical metrics + visualizations)
- ✅ Recommendations (persona-based)
- ✅ Documentation (all 3 guides load)

---

## 🔧 Troubleshooting

### App Won't Start

**Error**: `ModuleNotFoundError`

**Solution**: 
- Check `requirements.txt` has all dependencies
- Verify Python version in `runtime.txt` matches

**Error**: `FileNotFoundError` for demo data

**Solution**:
- Verify files exist in `usecases/` directory
- Check paths in `app.py` USE_CASES configuration

### Persona Evaluation Fails

**Error**: `OpenAI API error` or `Authentication failed`

**Solution**:
- Verify API key is set in Streamlit Cloud secrets
- Check API key is valid and has credits
- Test with personas disabled first

### Out of Memory

**Error**: `MemoryError` or app crashes during evaluation

**Solution**:
- Reduce `sample_size` in sidebar (try 50-100)
- Streamlit Cloud free tier has 1GB RAM limit
- Consider upgrading to paid tier for larger evaluations

### Slow Performance

**Issue**: App takes too long to load or evaluate

**Solution**:
- First load is always slower (cold start)
- Reduce sample size for faster evaluations
- Cache results using `st.cache_data` (future enhancement)

---

## 🔄 Continuous Deployment

### Automatic Updates

Streamlit Cloud automatically redeploys when you push to GitHub:

1. Make changes locally
2. Test locally: `streamlit run hexeval/ui/app.py`
3. Commit and push:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin hexeval
   ```
4. Streamlit Cloud auto-deploys (watch logs in dashboard)

### CI/CD Pipeline

GitHub Actions runs automatically on every push:

- **Test Suite**: Runs all tests
- **Code Quality**: Checks formatting and linting

If tests fail, fix issues before merging to main branch.

---

## 📊 Monitoring

### View Logs

1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click **"Manage app"** → **"Logs"**
4. Monitor for errors or warnings

### Usage Analytics

Streamlit Cloud provides basic analytics:
- Number of viewers
- App uptime
- Resource usage

Access via app settings in dashboard.

---

## 🎯 Best Practices

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make changes and test locally**

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Commit and push**:
   ```bash
   git push origin feature/new-feature
   ```

5. **Create Pull Request** on GitHub

6. **Wait for CI/CD checks** to pass

7. **Merge to main** → Auto-deploys to production

### Security

- ✅ Never commit API keys or secrets
- ✅ Use `.env` for local development
- ✅ Use Streamlit secrets for production
- ✅ Keep dependencies updated
- ✅ Monitor for security vulnerabilities

### Performance

- ✅ Use `@st.cache_data` for expensive computations
- ✅ Limit sample sizes for faster evaluations
- ✅ Optimize model loading (load once, reuse)
- ✅ Consider upgrading to paid tier for production

---

## 🆘 Getting Help

### Resources

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

### Common Issues

Check the [Streamlit Community Cloud FAQ](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/app-dependencies) for common deployment issues.

---

## 🎉 Success!

Your HEXEval app is now live and accessible to the world! 🚀

**Next Steps**:
- Share your app URL with stakeholders
- Add app URL to your README
- Consider adding more use case examples
- Collect user feedback for improvements

---

## 📝 Deployment Checklist

Use this checklist for future deployments:

- [ ] Security: API keys removed from code
- [ ] Testing: All tests pass locally
- [ ] GitHub: Changes pushed to repository
- [ ] CI/CD: GitHub Actions workflows pass
- [ ] Streamlit: App deployed successfully
- [ ] Secrets: OpenAI API key configured
- [ ] Verification: Both demo use cases work
- [ ] Documentation: README updated with live URL
- [ ] Monitoring: Logs checked for errors
- [ ] Performance: App loads within acceptable time

---

**Last Updated**: 2026-01-21
**Deployment Platform**: Streamlit Community Cloud (Free Tier)
**CI/CD**: GitHub Actions
