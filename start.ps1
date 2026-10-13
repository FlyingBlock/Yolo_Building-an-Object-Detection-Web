# ============================================================================
#  YOLO Object Detection Web App -- one-click launcher
#
#  Starts both halves of the app and opens the browser:
#     * FastAPI backend   (api.py)   -> http://127.0.0.1:8000  (+ /docs)
#     * Streamlit UI      (app.py)   -> http://localhost:8501
#
#  Normally you never call this file directly -- double-click "一键启动.bat".
#
#  Interpreter priority:
#     1. -PythonPath <python.exe>            (explicit override)
#     2. conda environment "yoloweb"          (Miniforge / Anaconda / Miniconda)
#     3. project .venv
#     4. system Python 3.9+
#  Step 2 means the launcher reproduces exactly what
#      conda activate yoloweb
#  does in a Miniforge Prompt -- no conda activation, shell hooks or two terminals
#  required.
#
#  Usage:
#      powershell -ExecutionPolicy Bypass -File start.ps1 [-ApiPort 8000]
#                 [-UiPort 8501] [-NoBrowser] [-NoApi] [-SetupOnly]
#                 [-Reinstall] [-Preload] [-PythonPath <python.exe>]
#                 [-CondaEnv yoloweb] [-NoConda]
# ============================================================================

[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [switch]$NoBrowser,
    [switch]$NoApi,
    [switch]$SetupOnly,
    [switch]$Reinstall,
    [switch]$Preload,
    [string]$PythonPath,
    [string]$CondaEnv = 'yoloweb',
    [switch]$NoConda
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- output ----
function Write-Head([string]$Text) { Write-Host ''; Write-Host "== $Text" -ForegroundColor Magenta }
function Write-Ok([string]$Text) { Write-Host "   [ok]   $Text" -ForegroundColor Green }
function Write-Info([string]$Text) { Write-Host "   ...    $Text" -ForegroundColor Gray }
function Write-Warn([string]$Text) { Write-Host "   [!]    $Text" -ForegroundColor Yellow }
function Write-Err([string]$Text) { Write-Host "   [x]    $Text" -ForegroundColor Red }

# ------------------------------------------------------------- project ------
$script:Root = $PSScriptRoot
if (-not $script:Root) { $script:Root = (Get-Location).Path }
$script:VenvDir = Join-Path $script:Root '.venv'
$script:VenvPython = Join-Path $script:VenvDir 'Scripts\python.exe'
$script:LogDir = Join-Path $script:Root 'logs'
$script:ApiOut = Join-Path $script:LogDir 'api.out.log'
$script:ApiErr = Join-Path $script:LogDir 'api.err.log'
$script:UiOut = Join-Path $script:LogDir 'ui.out.log'
$script:UiErr = Join-Path $script:LogDir 'ui.err.log'
$script:Processes = @()
$script:PyExe = $null
$script:CreatedVenv = $false

Write-Host ''
Write-Host '  ==========================================================' -ForegroundColor Cyan
Write-Host '    YOLO Object Detection Web App -- one-click launcher' -ForegroundColor Cyan
Write-Host '  ==========================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host "  Project : $script:Root"

# ------------------------------------------------------------ requirements --
$script:CoreDeps = @('fastapi', 'uvicorn', 'streamlit', 'multipart', 'requests', 'ultralytics', 'torch', 'cv2', 'PIL', 'numpy', 'pandas')

# Import name -> distribution name, used when a dependency must be installed by name.
$script:PackageNames = @{
    'fastapi'     = 'fastapi'
    'uvicorn'     = 'uvicorn[standard]'
    'streamlit'   = 'streamlit'
    'multipart'   = 'python-multipart'
    'requests'    = 'requests'
    'ultralytics' = 'ultralytics'
    'torch'       = 'torch'
    'cv2'         = 'opencv-python'
    'PIL'         = 'pillow'
    'numpy'       = 'numpy'
    'pandas'      = 'pandas'
}
$script:VersionSql = @'
import importlib.metadata as md
mods = ["fastapi", "uvicorn", "streamlit", "multipart", "requests", "ultralytics", "torch", "cv2", "PIL", "numpy", "pandas"]
dists = {"multipart": "python-multipart", "cv2": "opencv-python", "PIL": "pillow"}
for m in mods:
    try:
        __import__(m)
        try:
            v = md.version(dists.get(m, m))
        except Exception:
            v = "?"
        print(m + "\t" + v)
    except Exception:
        pass
'@

function Test-Dependencies([string]$Exe) {
    $rows = @()
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $rows }
    try {
        $raw = & $Exe -c $script:VersionSql 2>$null
    } catch {
        return $rows
    }
    foreach ($line in @($raw)) {
        if ($line -and $line.Contains("`t")) {
            $parts = $line.Split("`t")
            $rows += [pscustomobject]@{ Name = $parts[0]; Version = $parts[1] }
        }
    }
    return @($rows)
}

function Show-Versions([object[]]$Rows) {
    $index = @{}
    foreach ($r in $Rows) { $index[$r.Name] = $r.Version }
    $keys = @('torch', 'ultralytics', 'streamlit', 'fastapi', 'uvicorn', 'opencv-python') | Where-Object { $index.ContainsKey($_) }
    $text = @()
    foreach ($k in $keys) { $text += ("$k $($index[$k])") }
    if ($text.Count -gt 0) { Write-Info ($text -join ' | ') }
}

function Test-TcpPort([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connect = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($connect.AsyncWaitHandle.WaitOne(300)) {
            $client.EndConnect($connect)
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

# --------------------------------------------------------------- interpreter -
function Get-PythonVersion([string]$Exe) {
    try {
        $probe = & $Exe -c "import sys; print(sys.version_info[0]*100 + sys.version_info[1])" 2>$null
        $text = "$probe".Trim()
        if ($LASTEXITCODE -eq 0 -and $text -match '^\d+$') { return [int]$text }
    } catch { }
    return 0
}

function Resolve-SystemPython {
    $candidates = @()

    # `py -0p` lists the real interpreters registered with the launcher. The launcher
    # stub itself (py.exe) cannot run `-m venv`, so collect concrete python.exe paths.
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            foreach ($line in @(& $pyLauncher.Source -0p 2>$null)) {
                if (-not $line) { continue }
                $match = [regex]::Match("$line", '([A-Za-z]:\\[^\r\n]*?python(?:w)?\.exe)')
                if ($match.Success) { $candidates += $match.Groups[1].Value.Trim() }
            }
        } catch { }
    }

    foreach ($name in @('python.exe', 'python3.exe')) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found -and $found.Source) { $candidates += $found.Source }
    }

    $fallback = $null
    $checked = @{}
    foreach ($candidate in $candidates) {
        $key = $candidate.ToLowerInvariant()
        if ($checked.ContainsKey($key)) { continue }
        $checked[$key] = $true
        $version = Get-PythonVersion $candidate
        if ($version -ge 309) { return $candidate }
        if ($version -gt 0 -and -not $fallback) { $fallback = $candidate }
    }
    return $fallback
}

function Resolve-CondaRoot {
    # Anaconda-family interpreters live in <root>\envs\<name>, so the root has to be
    # found before an environment can be used without `conda activate`. `where.exe`
    # runs here because PATH itself is never modified by this launcher.
    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
        $candidate = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
        if (Test-Path -LiteralPath (Join-Path $candidate 'envs')) { return $candidate }
    }
    if ($env:CONDA_ROOT -and (Test-Path -LiteralPath (Join-Path $env:CONDA_ROOT 'envs'))) {
        return $env:CONDA_ROOT
    }

    try {
        $found = & where.exe conda.exe 2>$null
        foreach ($line in @($found)) {
            if (-not $line) { continue }
            $candidate = Split-Path -Parent (Split-Path -Parent "$line".Trim())
            if (Test-Path -LiteralPath (Join-Path $candidate 'envs')) { return $candidate }
        }
    } catch { }

    $roots = @()
    # NOTE: do not use $home -- PowerShell variables are case-insensitive and $HOME
    # is read-only, so assigning to it throws.
    $userHome = $env:USERPROFILE
    if (-not $userHome) { $userHome = "$env:HOMEDRIVE$env:HOMEPATH" }
    foreach ($leaf in @('miniforge3', 'Miniforge3', 'mambaforge', 'Mambaforge', 'miniconda3', 'Miniconda3', 'anaconda3', 'Anaconda3', 'miniforge', 'conda')) {
        foreach ($base in @($userHome, 'C:\', 'D:\', 'C:\ProgramData', 'D:\ProgramData', (Join-Path $userHome 'AppData\Local'))) {
            if ($base) { $roots += (Join-Path $base $leaf) }
        }
    }
    foreach ($root in $roots) {
        if ($root -and (Test-Path -LiteralPath (Join-Path $root 'envs'))) { return $root }
    }
    return $null
}

function Resolve-CondaPython {
    # Returns the python.exe of the named conda env, or $null when the env does not
    # exist. Envs are matched case-insensitively the way conda itself matches them.
    if ($NoConda) { return $null }
    $name = "$CondaEnv".Trim()
    if (-not $name) { return $null }

    $root = Resolve-CondaRoot
    if (-not $root) { return $null }

    $direct = Join-Path $root "envs\$name\python.exe"
    if (Test-Path -LiteralPath $direct) { return $direct }

    $envsDir = Join-Path $root 'envs'
    foreach ($dir in @(Get-ChildItem -LiteralPath $envsDir -Directory -ErrorAction SilentlyContinue)) {
        if ($dir.Name -ieq $name) {
            $exe = Join-Path $dir.FullName 'python.exe'
            if (Test-Path -LiteralPath $exe) { return $exe }
        }
    }
    return $null
}

function Use-CondaEnvironment([string]$Exe) {
    # Reproduce the part of `conda activate` that this launcher actually depends on:
    # subprocesses (uvicorn / streamlit, including `--reload` children) inherit these
    # and therefore run with the conda env on PATH.
    $prefix = Split-Path -Parent $Exe
    $env:CONDA_PREFIX = $prefix
    $env:CONDA_DEFAULT_ENV = Split-Path -Leaf $prefix
    $env:CONDA_SHLVL = '1'
    $parts = @($prefix, (Join-Path $prefix 'Library\mingw-w64\bin'), (Join-Path $prefix 'Library\usr\bin'),
        (Join-Path $prefix 'Library\bin'), (Join-Path $prefix 'Scripts'), (Join-Path $prefix 'bin'))
    $existing = @()
    if ($env:PATH) { $existing = $env:PATH -split ';' }
    $keep = @($existing | Where-Object { $_ -and ($parts -notcontains $_) })
    $env:PATH = (@($parts) + $keep) -join ';'
}

function Resolve-Python {
    # Returns @{ Exe = <path>; Source = <label> }; Exe is $null when nothing usable exists.
    if ($PythonPath) {
        if (Test-Path -LiteralPath $PythonPath) {
            return @{ Exe = $PythonPath; Source = 'command line (-PythonPath)' }
        }
        Write-Warn "指定的 Python 不存在，改为自动检测：$PythonPath"
    }

    $condaExe = Resolve-CondaPython
    if ($condaExe) {
        return @{ Exe = $condaExe; Source = "conda 环境 '$CondaEnv'" }
    }

    if (Test-Path -LiteralPath $script:VenvPython) {
        return @{ Exe = $script:VenvPython; Source = '项目虚拟环境 .venv' }
    }

    $systemExe = Resolve-SystemPython
    if ($systemExe) { return @{ Exe = $systemExe; Source = '系统 Python' } }
    return @{ Exe = $null; Source = $null }
}

function New-Venv([string]$BasePython, [switch]$UseSystemPackages) {
    Write-Head '首次运行：创建虚拟环境 .venv'
    if (Test-Path -LiteralPath $script:VenvDir) {
        Remove-Item -LiteralPath $script:VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($UseSystemPackages) {
        Write-Info '复用系统 Python 中已有的 torch / ultralytics，避免重复下载数 GB 依赖。'
        & $BasePython -m venv --system-site-packages $script:VenvDir
    } else {
        Write-Info 'torch 体积较大，安装可能需要几分钟，请耐心等待…'
        & $BasePython -m venv $script:VenvDir
    }
    if (-not (Test-Path -LiteralPath $script:VenvPython)) {
        throw "虚拟环境创建失败（$script:VenvDir）。"
    }
    $script:CreatedVenv = $true
    Write-Ok ".venv 已创建：$script:VenvDir"
}

function Get-MissingDeps([string]$Target) {
    $have = @((Test-Dependencies $Target) | ForEach-Object { $_.Name })
    return @($script:CoreDeps | Where-Object { $have -notcontains $_ })
}

function Install-Requirements([string]$Exe, [string]$BasePython) {
    Write-Head '安装项目依赖'
    if ($Exe -eq $script:VenvPython) {
        Write-Info '初始化 .venv 中的 pip…'
        if ($BasePython) {
            # `python -m venv` cannot always bootstrap pip (blocked/patched ensurepip);
            # let the base interpreter install pip *into* the venv instead.
            & $BasePython -m pip --python $script:VenvPython install --upgrade pip 2>$null | Out-Null
        }
    }
    & $Exe -m pip install --upgrade pip | Out-Null

    # Install only what is actually missing. When the venv was created with
    # --system-site-packages, torch and friends are already importable and a full
    # `-r requirements.txt` run would re-download several GB for nothing.
    $stillMissing = Get-MissingDeps $Exe
    if ($stillMissing.Count -eq 0) {
        Write-Ok '所需依赖已全部可用，无需安装。'
        return
    }

    Write-Info ("需要安装：" + ($stillMissing -join ', '))
    & $Exe -m pip install -r (Join-Path $script:Root 'requirements.txt')
    $status = $LASTEXITCODE

    # Fall back to installing the still-missing distributions by name, so a hiccup
    # part-way through requirements.txt does not leave the app unable to start.
    $stillMissing = Get-MissingDeps $Exe
    if ($stillMissing.Count -gt 0) {
        $packages = @($stillMissing | ForEach-Object { $script:PackageNames[$_] })
        Write-Warn ("改用逐项安装：" + ($packages -join ', '))
        & $Exe -m pip install @packages
        $status = $LASTEXITCODE
    }

    if ($status -ne 0) { throw 'pip install 失败，请检查网络或上面的报错信息。' }
    Write-Ok '依赖安装完成。'
}

function Test-VenvReady {
    if (-not (Test-Path -LiteralPath $script:VenvPython)) { return $false }
    try {
        & $script:VenvPython -c "import streamlit, uvicorn, fastapi" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# -------------------------------------------------------------- environment --
function Initialize-Environment {
    $resolved = Resolve-Python
    $exe = $resolved.Exe
    if (-not $exe) {
        throw "没有找到可用的 Python。请安装 Python 3.9+，或创建 conda 环境 '$CondaEnv'，再重新运行本启动器。"
    }

    $script:PySource = $resolved.Source
    $script:UsingConda = ($resolved.Source -like 'conda*')
    if ($script:UsingConda) {
        Use-CondaEnvironment $exe
        Write-Info "来源    : $($resolved.Source)（已注入 conda 环境变量，无需 Miniforge Prompt）"
    } else {
        Write-Info "来源    : $($resolved.Source)"
    }
    Write-Info "Python  : $exe"
    Write-Info "工作目录: $script:Root"

    if (-not (Test-Path -LiteralPath $script:LogDir)) {
        New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
    }

    # A half-created or stale .venv would otherwise break every later step: if the
    # venv interpreter cannot run at all, fall back to a system interpreter.
    if ($exe -eq $script:VenvPython -and (Get-PythonVersion $exe) -eq 0) {
        $fallbackExe = Resolve-SystemPython
        if (-not $fallbackExe) {
            throw "项目内的 .venv 似乎已损坏，且没有找到其它可用的 Python。请删除 .venv 目录后重新运行本启动器。"
        }
        Write-Warn ".venv 中的 Python 无法运行，改用系统解释器：$fallbackExe"
        Write-Info "（如果要重建虚拟环境，请运行：start.ps1 -Reinstall）"
        $exe = $fallbackExe
    }

    $rows = Test-Dependencies $exe
    $have = @($rows | ForEach-Object { $_.Name })
    $missing = @($script:CoreDeps | Where-Object { $have -notcontains $_ })
    $installedVenv = ($exe -eq $script:VenvPython)

    if ($missing.Count -eq 0) {
        Show-Versions $rows
        Write-Ok '依赖齐全，直接启动。'
        $script:PyExe = $exe
        return
    }

    Write-Warn ("缺少依赖：" + ($missing -join ', '))

    if ($installedVenv) {
        Install-Requirements $exe $exe
    } elseif ($script:UsingConda) {
        # The interpreter came from a conda env, so that env is the environment the
        # user wants -- repair it in place instead of quietly building a second one.
        Install-Requirements $exe $null
        $script:PyExe = $exe
        Show-Versions (Test-Dependencies $script:PyExe)
        return
    } elseif ($Reinstall) {
        New-Venv $exe
        Install-Requirements $script:VenvPython $exe
    } else {
        Write-Host ''
        Write-Host '   检测到当前 Python 中缺少部分依赖。请选择处理方式：'
        Write-Host '     Y = 新建项目内虚拟环境 .venv 并安装缺失依赖（推荐）'
        Write-Host '     N = 不安装，直接用当前 Python 启动（若缺模块会启动失败）'
        $answer = Read-Host '   继续安装? [Y/N]'
        if ($answer -match '^(y|yes|是|好)$') {
            # Reuse an already-complete system install (e.g. working CUDA torch)
            # instead of re-downloading multi-GB wheels into a fresh venv.
            $baseHeavy = @('torch', 'ultralytics', 'cv2', 'numpy', 'pandas')
            $baseHasHeavy = @($baseHeavy | Where-Object { $have -contains $_ }).Count -eq $baseHeavy.Count
            if (-not $installedVenv -and $baseHasHeavy) {
                New-Venv $exe -UseSystemPackages
            } else {
                New-Venv $exe
            }
            if (-not (Test-Path -LiteralPath $script:VenvPython)) {
                throw '虚拟环境创建失败。'
            }
            try {
                Install-Requirements $script:VenvPython $exe
            } catch {
                if (Test-VenvReady) {
                    Write-Warn "安装过程中出现问题（$($_.Exception.Message)），但依赖已可用，继续启动。"
                } else {
                    throw
                }
            }
            if (-not (Test-VenvReady)) {
                throw "依赖未能安装完整。请手动执行：$script:VenvPython -m pip install -r requirements.txt"
            }
        } else {
            Write-Warn '已跳过安装。若启动报缺少模块，请重新运行并选择 Y，或手动执行：'
            Write-Info "  $exe -m pip install -r requirements.txt"
        }
    }

    # Prefer the project venv if one now exists, otherwise stay on the detected interpreter.
    # An explicitly chosen conda env always wins over .venv.
    if ($script:UsingConda) {
        $script:PyExe = $exe
    } elseif (Test-Path -LiteralPath $script:VenvPython) {
        $script:PyExe = $script:VenvPython
    } else {
        Write-Warn '继续使用当前 Python 启动（未创建 .venv）。'
        $script:PyExe = $exe
    }
    Show-Versions (Test-Dependencies $script:PyExe)
}

# ------------------------------------------------------------- lifecycle ----
function Add-Child([string]$Name, [System.Diagnostics.Process]$Process) {
    $script:Processes += [pscustomobject]@{ Name = $Name; Process = $Process }
}

function Stop-All {
    if (-not $script:Processes -or $script:Processes.Count -eq 0) { return }
    Write-Host ''
    Write-Info '正在停止服务…'
    foreach ($child in $script:Processes) {
        try {
            if ($child.Process -and -not $child.Process.HasExited) {
                Stop-Process -Id $child.Process.Id -Force -ErrorAction SilentlyContinue
            }
        } catch { }
    }
    $script:Processes = @()
}

function Start-Api {
    Write-Head "启动后端 API（uvicorn，端口 $ApiPort）"
    if (-not (Test-Path -LiteralPath $script:LogDir)) {
        New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
    }
    $env:PYTHONUNBUFFERED = '1'
    $env:PRELOAD_MODEL = if ($Preload) { '1' } else { '0' }

    $argumentList = @('-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', "$ApiPort")
    $process = Start-Process -FilePath $script:PyExe -ArgumentList $argumentList `
        -WorkingDirectory $script:Root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $script:ApiOut -RedirectStandardError $script:ApiErr
    Add-Child 'API' $process
    Write-Info "日志：$script:ApiOut"
}

function Start-Ui {
    Write-Head "启动前端 UI（Streamlit，端口 $UiPort）"

    $env:API_URL = "http://127.0.0.1:$ApiPort"
    $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
    $env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE = 'false'
    $env:STREAMLIT_SERVER_HEADLESS = 'true'

    $argumentList = @(
        '-m', 'streamlit', 'run', 'app.py',
        '--server.port', "$UiPort",
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.fileWatcherType', 'auto'
    )
    $process = Start-Process -FilePath $script:PyExe -ArgumentList $argumentList `
        -WorkingDirectory $script:Root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $script:UiOut -RedirectStandardError $script:UiErr
    Add-Child 'UI' $process
    Write-Info "日志：$script:UiOut"
}

function Wait-Http([string]$Url, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-TcpPort ([uri]$Url).Port) {
            try {
                $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return $true }
            } catch {
                # Server is listening but not answering yet -- keep waiting.
            }
        }
        foreach ($child in $script:Processes) {
            if ($child.Process.HasExited) { return $false }
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Show-LogTail([string]$Path, [string]$Label) {
    if (Test-Path -LiteralPath $Path) {
        $tail = @(Get-Content -LiteralPath $Path -Tail 20 -ErrorAction SilentlyContinue)
        if ($tail.Count -gt 0) {
            Write-Host ''
            Write-Host "   ---- $Label（最后 20 行）----" -ForegroundColor DarkGray
            foreach ($line in $tail) { Write-Host "   $line" -ForegroundColor DarkGray }
            Write-Host ''
        }
    }
}

# ---------------------------------------------------------------- main ------
try {
    Set-Location -LiteralPath $script:Root

    foreach ($required in @('api.py', 'app.py', 'requirements.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $script:Root $required))) {
            throw "在 $script:Root 下找不到 $required —— 请把启动器放在项目根目录运行。"
        }
    }

    Initialize-Environment

    if ($SetupOnly) {
        Write-Host ''
        Write-Ok '环境准备完成（-SetupOnly）。'
        Write-Host ''
        exit 0
    }

    $startedApi = $false
    $startedUi = $false

    if ($NoApi) {
        Write-Warn '按 -NoApi 跳过后端：请在侧边栏把“推理后端”切换为“本地推理（进程内）”。'
    } elseif (Test-TcpPort $ApiPort) {
        Write-Warn "端口 $ApiPort 已被占用，将复用已在运行的服务（若它不是本项目的 API，请改用 -ApiPort <其它端口>）。"
    } else {
        Start-Api
        $startedApi = $true
    }

    if (Test-TcpPort $UiPort) {
        Write-Warn "端口 $UiPort 已被占用，将复用已在运行的服务（若它不是本项目的界面，请改用 -UiPort <其它端口>）。"
    } else {
        Start-Ui
        $startedUi = $true
    }

    Write-Host ''
    $apiUrl = "http://127.0.0.1:$ApiPort"
    $uiUrl = "http://localhost:$UiPort"

    $apiOk = $true
    $uiOk = $true

    if ($NoApi) {
        Write-Info '已按 -NoApi 跳过后端检查。'
    } elseif ($startedApi) {
        Write-Info '等待后端就绪…'
        $apiOk = Wait-Http "$apiUrl/health" 120
        if ($apiOk) { Write-Ok "后端就绪：$apiUrl/health" } else { Write-Warn "后端未在 120 秒内就绪，请查看日志：$script:ApiOut" }
    } else {
        Write-Ok "后端已在线：$apiUrl/health"
    }

    if ($startedUi) {
        Write-Info '等待前端就绪…'
        $uiOk = Wait-Http "$uiUrl/" 120
        if ($uiOk) { Write-Ok "前端就绪：$uiUrl" } else { Write-Warn "前端未在 120 秒内就绪，请查看日志：$script:UiOut" }
    } else {
        Write-Ok "前端已在线：$uiUrl"
    }

    $failed = @($script:Processes | Where-Object { $_.Process.HasExited })
    foreach ($child in $failed) {
        Write-Err "$($child.Name) 进程已退出（退出码 $($child.Process.ExitCode)）。"
    }

    Write-Host ''
    Write-Host '  ==========================================================' -ForegroundColor Cyan
    Write-Host '   服务已启动' -ForegroundColor Cyan
    Write-Host '  ==========================================================' -ForegroundColor Cyan
    Write-Host "   界面      : $uiUrl"
    if (-not $NoApi) {
        Write-Host "   接口文档  : $apiUrl/docs"
        Write-Host "   健康检查  : $apiUrl/health"
    }
    Write-Host "   日志目录  : $script:LogDir"
    if ($script:PySource) { Write-Host "   解释器    : $script:PyExe（$script:PySource）" }
    Write-Host ''
    Write-Host '   首次检测会下载权重并预热 GPU，可能需要十几秒。' -ForegroundColor DarkGray
    Write-Host '   在这个窗口按 Ctrl+C 即可停止全部服务。' -ForegroundColor DarkGray

    if (-not $NoBrowser) {
        $target = if ($uiOk) { $uiUrl } else { $apiUrl }
        try {
            Start-Process $target | Out-Null
            Write-Info "已打开浏览器：$target"
        } catch {
            Write-Warn "无法自动打开浏览器，请手动访问 $uiUrl"
        }
    }

    if ($failed.Count -gt 0) {
        foreach ($child in $failed) {
            $log = if ($child.Name -eq 'API') { $script:ApiErr } else { $script:UiErr }
            Show-LogTail $log $child.Name
        }
        throw '有服务未能启动，已停止其余服务。'
    }

    Write-Host ''
    Write-Info '监视中…（Ctrl+C 停止）'
    try {
        while ($true) {
            Start-Sleep -Seconds 2
            foreach ($child in $script:Processes) {
                if ($child.Process.HasExited) {
                    Write-Err "$($child.Name) 意外退出（退出码 $($child.Process.ExitCode)）。"
                    throw "$($child.Name) 已停止。"
                }
            }
        }
    } finally {
        Stop-All
    }
} catch {
    Write-Host ''
    Write-Err $_.Exception.Message
    Stop-All
    Write-Host ''
    Write-Host "   如果问题持续，可以试试重新创建虚拟环境：" -ForegroundColor DarkGray
    Write-Host "     powershell -ExecutionPolicy Bypass -File start.ps1 -Reinstall -SetupOnly" -ForegroundColor DarkGray
    exit 1
}
