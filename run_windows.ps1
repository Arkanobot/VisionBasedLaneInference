<#
.SYNOPSIS
  Launch the lane-inference web application (Windows/PowerShell).

.USAGE
  .\run.ps1              serve on this machine only
  .\run.ps1 -Lan         also reachable from a phone on the same wifi
  .\run.ps1 -Dev         Vite dev server with hot reload, API proxied
  .\run.ps1 -Port 9000   use a custom port

  Everything runs from inside this directory; nothing is installed globally.
#>

param(
    [switch]$Lan,
    [switch]$Dev,
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8000 })
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$HostAddr = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
$VenvPython = Join-Path ".venv" "Scripts\python.exe"

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv here. Create one and install lane_inference/requirements.txt"
    exit 1
}

if ($Dev) {
    Write-Host "API   -> http://127.0.0.1:8000"
    Write-Host "Front -> http://localhost:5173  (hot reload)"

    $apiProc = Start-Process -FilePath $VenvPython `
        -ArgumentList "-m", "uvicorn", "server.api:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
        -PassThru -NoNewWindow

    try {
        Push-Location "web"
        npm run dev
    }
    finally {
        Pop-Location
        if ($apiProc -and -not $apiProc.HasExited) {
            Stop-Process -Id $apiProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    exit 0
}

if (-not (Test-Path "web\dist")) {
    Write-Host "Building the frontend (first run only)..."
    Push-Location "web"
    npm install --silent
    npm run build
    Pop-Location
}

if ($HostAddr -eq "0.0.0.0") {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.InterfaceAlias -match '^(Wi-?Fi|Ethernet)' -and
            $_.IPAddress -notlike "169.254.*"
        } |
        Select-Object -First 1 -ExpandProperty IPAddress)
    if (-not $ip) { $ip = "127.0.0.1" }

    Write-Host ""
    Write-Host "  ----------------------------------------------------"
    Write-Host "   On this PC : http://localhost:$Port"
    Write-Host "   On a phone : http://${ip}:$Port      (same wifi)"
    Write-Host ""
    Write-Host "   Uploads and every tab work over plain HTTP."
    Write-Host "   The camera button will not -- browsers only allow"
    Write-Host "   camera access over HTTPS or on localhost."
    Write-Host "  ----------------------------------------------------"
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "  http://localhost:$Port"
    Write-Host ""
}

& $VenvPython -m uvicorn server.api:app --host $HostAddr --port $Port