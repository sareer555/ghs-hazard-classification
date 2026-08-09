# =============================================================================
# share_app.ps1 - put the GHS screening app on a temporary public URL
# =============================================================================
#
# WHAT THIS DOES
#   Starts the Streamlit app (if it is not already running) and opens a
#   Cloudflare "quick tunnel" to it. Cloudflare hands back an address like
#       https://random-words-1234.trycloudflare.com
#   which works from any device, anywhere - including a phone on mobile data.
#
# WHAT TO KNOW BEFORE YOU RUN IT
#   * While the tunnel is open, ANYONE who has the link can use the app.
#     There is no password. Only share it with people you mean to.
#   * Every prediction runs on THIS computer and queries PubChem from THIS
#     internet connection.
#   * The link dies the moment you close this window, and a new run gives a
#     different address. It is meant for showing someone your work, not for
#     leaving online.
#   * No Cloudflare account, no admin rights and no firewall changes needed -
#     the tunnel dials out from your machine rather than accepting an
#     incoming connection.
#
# USAGE
#   Start sharing:   powershell -ExecutionPolicy Bypass -File share_app.ps1
#   Stop sharing:    press Ctrl+C, or just close the window
#
# =============================================================================

$ErrorActionPreference = "Continue"
$AppDir      = "D:\GHS_Project"
$Streamlit   = "D:\GHS_Project\.venv\Scripts\streamlit.exe"
$Cloudflared = "C:\Users\hafiz\cloudflared\cloudflared.exe"
$Port        = 8501

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " GHS Chemical Hazard Screening - share over the internet" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# ---- 1. make sure the app itself is up --------------------------------------
function Test-App {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$Port/_stcore/health" `
             -UseBasicParsing -TimeoutSec 5
        return $r.Content -match "ok"
    } catch { return $false }
}

if (Test-App) {
    Write-Host "[1/3] Streamlit app already running on port $Port" -ForegroundColor Green
} else {
    Write-Host "[1/3] Starting the Streamlit app..." -ForegroundColor Yellow
    Start-Process -FilePath $Streamlit `
        -ArgumentList "run", "$AppDir\app.py" `
        -WorkingDirectory $AppDir `
        -RedirectStandardOutput "$AppDir\logs\streamlit.log" `
        -RedirectStandardError  "$AppDir\logs\streamlit.err" `
        -WindowStyle Hidden

    $ready = $false
    foreach ($i in 1..25) {
        Start-Sleep -Seconds 2
        if (Test-App) { $ready = $true; break }
    }
    if ($ready) {
        Write-Host "      app is up" -ForegroundColor Green
    } else {
        Write-Host "      the app did not start. Check logs\streamlit.err" -ForegroundColor Red
        exit 1
    }
}

# ---- 2. sanity-check the tunnel binary --------------------------------------
if (-not (Test-Path $Cloudflared)) {
    Write-Host "[2/3] cloudflared not found at $Cloudflared" -ForegroundColor Red
    Write-Host "      Download it from:" -ForegroundColor Yellow
    Write-Host "      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    exit 1
}
Write-Host "[2/3] cloudflared found" -ForegroundColor Green

# ---- 3. open the tunnel ------------------------------------------------------
Write-Host "[3/3] Opening the tunnel - your public link appears below" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Look for a line ending in .trycloudflare.com" -ForegroundColor Cyan
Write-Host "  Send THAT link to your friend." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Anyone with the link can use the app while this window is open." -ForegroundColor Yellow
Write-Host "  Press Ctrl+C, or close this window, to stop sharing." -ForegroundColor Yellow
Write-Host ""
Write-Host "--------------------------------------------------------------" -ForegroundColor DarkGray

# --url makes this a "quick tunnel": no Cloudflare account is involved and
# nothing is registered against your name. The address is random and single-use.
& $Cloudflared tunnel --url "http://localhost:$Port" --no-autoupdate

Write-Host ""
Write-Host "Tunnel closed. The public link no longer works." -ForegroundColor Green
Write-Host "The app itself is still running locally at http://localhost:$Port" -ForegroundColor Gray
