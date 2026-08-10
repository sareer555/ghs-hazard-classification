# =============================================================================
# push_to_github.ps1 - upload this project to GitHub
# =============================================================================
#
# HOW TO RUN IT
#   Right-click this file in File Explorer and choose "Run with PowerShell".
#   If that option is missing, open File Explorer at D:\GHS_Project, click the
#   address bar, type  powershell  and press Enter, then run:
#
#       powershell -ExecutionPolicy Bypass -File push_to_github.ps1
#
# BEFORE YOU RUN IT
#   Create an EMPTY repository at https://github.com/new
#     - Name:  ghs-hazard-classification
#     - Do NOT tick "Add a README file"
#     - Do NOT add a .gitignore or a licence
#   You already have all three; letting GitHub create them causes a conflict.
#
# WHAT IT DOES
#   Points your local repository at GitHub and uploads every commit. Your files
#   are not modified. Running it twice is harmless.
# =============================================================================

$ErrorActionPreference = "Continue"
$Project  = "D:\GHS_Project"
$Username = "sareer555"
$RepoName = "ghs-hazard-classification"
$RemoteUrl = "https://github.com/$Username/$RepoName.git"

Set-Location $Project

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " Push the GHS project to GitHub" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1. show what is about to be uploaded ----------------------------------
$commits = git log --oneline 2>$null
if (-not $commits) {
    Write-Host "No commits found. Nothing to push." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}
Write-Host "Commits ready to upload:" -ForegroundColor Green
git log --pretty=format:"   %h  %s" | Out-Host
Write-Host ""

$files = (git ls-files).Count
Write-Host "Files tracked: $files" -ForegroundColor Green
Write-Host ""

# ---- 2. confirm the GitHub repository exists -------------------------------
Write-Host "This will upload to:" -ForegroundColor Yellow
Write-Host "   $RemoteUrl" -ForegroundColor White
Write-Host ""
Write-Host "That repository must already exist and be EMPTY." -ForegroundColor Yellow
Write-Host "Create it at https://github.com/new if you have not yet." -ForegroundColor Yellow
Write-Host ""
$go = Read-Host "Has the empty repository been created? (y/n)"
if ($go -ne "y") {
    Write-Host ""
    Write-Host "Create it first, then run this script again." -ForegroundColor Cyan
    Read-Host "Press Enter to close"
    exit 0
}

# ---- 3. point the local repository at GitHub -------------------------------
$existing = git remote get-url origin 2>$null
if ($existing) {
    Write-Host ""
    Write-Host "A remote is already set: $existing" -ForegroundColor Gray
    if ($existing -ne $RemoteUrl) {
        git remote set-url origin $RemoteUrl
        Write-Host "Updated it to $RemoteUrl" -ForegroundColor Green
    }
} else {
    git remote add origin $RemoteUrl
    Write-Host ""
    Write-Host "Remote added." -ForegroundColor Green
}

# ---- 4. push ----------------------------------------------------------------
Write-Host ""
Write-Host "Uploading..." -ForegroundColor Yellow
Write-Host ""
Write-Host "You will be asked to sign in." -ForegroundColor Cyan
Write-Host "  Username: $Username" -ForegroundColor White
Write-Host "  Password: use a PERSONAL ACCESS TOKEN, not your GitHub password." -ForegroundColor White
Write-Host "            GitHub stopped accepting passwords for git in 2021." -ForegroundColor Gray
Write-Host "            Create one at:" -ForegroundColor Gray
Write-Host "            GitHub -> Settings -> Developer settings ->" -ForegroundColor Gray
Write-Host "            Personal access tokens -> Tokens (classic) ->" -ForegroundColor Gray
Write-Host "            Generate new token, tick the 'repo' scope." -ForegroundColor Gray
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Green
    Write-Host " Uploaded successfully" -ForegroundColor Green
    Write-Host "==============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host " Your repository:" -ForegroundColor Cyan
    Write-Host "   https://github.com/$Username/$RepoName" -ForegroundColor White
    Write-Host ""
    Write-Host " Next, to get the DOI your paper needs:" -ForegroundColor Cyan
    Write-Host "   1. Sign in at zenodo.org using your GitHub account" -ForegroundColor White
    Write-Host "   2. Settings -> GitHub -> switch this repository On" -ForegroundColor White
    Write-Host "   3. On GitHub: Releases -> Create a new release -> tag v1.0.0" -ForegroundColor White
    Write-Host "   4. Zenodo mints a DOI automatically - cite it in the paper" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "The push did not succeed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Common causes:" -ForegroundColor Yellow
    Write-Host "  * 'Repository not found'   - the repo does not exist yet, or the" -ForegroundColor Gray
    Write-Host "                               name is spelled differently" -ForegroundColor Gray
    Write-Host "  * 'Authentication failed'  - you used your password; use a" -ForegroundColor Gray
    Write-Host "                               personal access token instead" -ForegroundColor Gray
    Write-Host "  * 'Updates were rejected'  - the GitHub repo is not empty; it was" -ForegroundColor Gray
    Write-Host "                               created with a README or licence" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to close"
