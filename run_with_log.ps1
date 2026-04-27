# Uso (na raiz do repo):
#   .\run_with_log.ps1
#   .\run_with_log.ps1 train_exhaustive.py --profile cuda_16gb
#
# Grava o log em UTF-8 (com BOM) para emojis/acentos nao corromperem. Tee-Object em Windows
# usa o encoding do sistema e mexe UTF-8 do Python.
#
# Saída: terminal + ficheiro out-<yyyyMMdd-HHmmss>.txt no diretório atual

param(
    [Parameter(Mandatory = $false, ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Continue"
try { chcp 65001 | Out-Null } catch { }  # UTF-8 na consola (cmd antigo)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path (Get-Location) "out-$ts.txt"

$py = Join-Path $PSScriptRoot "venv_fs2\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Nao encontrei: $py  (cria/activa venv_fs2 ou ajusta o caminho no script)" -ForegroundColor Red
    exit 1
}

if ($null -eq $RemainingArgs -or $RemainingArgs.Count -eq 0) {
    $RemainingArgs = @("train_exhaustive.py", "--profile", "cuda_16gb")
}

$scriptPath = $RemainingArgs[0]
if ($scriptPath -notmatch '\.py$') {
    Write-Host "Primeiro argumento deve ser o .py (ex: train_exhaustive.py)" -ForegroundColor Yellow
}

$scriptFull = Join-Path $PSScriptRoot $scriptPath
$rest = @()
if ($RemainingArgs.Count -gt 1) {
    $rest = $RemainingArgs[1..($RemainingArgs.Count - 1)]
}

Write-Host "Log: $logFile" -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# UTF-8 com BOM: Notepad/VS abrem bem; evita mojibake (ƒÜÇ) face a Tee-Object em ANSI
$utf8Bom = New-Object System.Text.UTF8Encoding $true
$writer = [System.IO.StreamWriter]::new($logFile, $false, $utf8Bom)

function Convert-PipelineObjectToLine {
    param($o)
    if ($null -eq $o) { return "" }
    if ($o -is [string]) { return $o }
    # stderr (2>&1) vira ErrorRecord; ToString() mostra "RemoteException" em vez do texto
    if ($o -is [System.Management.Automation.ErrorRecord]) {
        $m = $o.Exception.Message
        if ($m) { return $m }
        if ($o.ErrorDetails -and $o.ErrorDetails.Message) { return $o.ErrorDetails.Message }
    }
    return $o.ToString()
}

$exit = 0
try {
    & $py $scriptFull @rest 2>&1 | ForEach-Object {
        $line = Convert-PipelineObjectToLine -o $_
        [Console]::WriteLine($line)
        $writer.WriteLine($line)
        $writer.Flush()
    }
    if ($null -ne $LASTEXITCODE) { $exit = $LASTEXITCODE }
} catch {
    $e = $_.ToString()
    [Console]::WriteLine($e)
    $writer.WriteLine($e)
    $exit = 1
} finally {
    $writer.Close()
}

if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { $exit = $LASTEXITCODE }
exit $exit
