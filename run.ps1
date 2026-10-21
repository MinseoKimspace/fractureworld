$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $taskPython) {
    & $taskPython -m fracture_world @args
} else {
    & uv run --locked --python 3.11 python -m fracture_world @args
}
exit $LASTEXITCODE
