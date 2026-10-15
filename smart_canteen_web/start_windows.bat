@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"


echo [1/3] 正在检查 Python...
set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
  if not defined PYTHON_CMD (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
  )
)

if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo 未找到 Python。请先安装 Python 3.10 至 3.12，并勾选“Add Python to PATH”。
  pause
  exit /b 1
)

%PYTHON_CMD% scripts\bootstrap.py %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo 启动失败，请查看上方提示，并参阅“新设备使用说明.md”。
pause
exit /b 1
