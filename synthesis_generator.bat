@echo off
chcp 65001 >nul
echo 正在啟動合成數據生成器 GUI...

REM ==========================================
REM  Conda 環境設定 (請依據您的電腦修改此處)
REM ==========================================

REM 1. 設定您的 Conda 環境名稱 (看您是用 base 還是有建立特定環境)
REM 根據您的截圖，您的環境名稱似乎是 "ballshift"
set CONDA_ENV_NAME=sam-3d

REM 2. 設定 Miniconda/Anaconda 的 activate.bat 路徑
REM 常見路徑如下，請確認您的實際安裝位置：
REM C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat
REM C:\ProgramData\miniconda3\Scripts\activate.bat
REM C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat
set CONDA_ACTIVATE_PATH=C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat

REM ==========================================

REM 1. 嘗試啟動環境
if exist "%CONDA_ACTIVATE_PATH%" (
    echo 偵測到 Conda路徑，正在啟動環境 [%CONDA_ENV_NAME%] ...
    call "%CONDA_ACTIVATE_PATH%" %CONDA_ENV_NAME%
) else (
    echo [提示] 未找到指定路徑的 Conda activate.bat，嘗試尋找專案內 venv...
    
    if exist ".venv\Scripts\activate.bat" (
        echo 偵測到 .venv，正在啟動虛擬環境...
        call ".venv\Scripts\activate.bat"
    ) else if exist "venv\Scripts\activate.bat" (
        echo 偵測到 venv，正在啟動虛擬環境...
        call "venv\Scripts\activate.bat"
    ) else (
        echo [警告] 未偵測到虛擬環境，將使用系統全域 Python...
    )
)

REM 2. 執行 Python 程式
echo.
echo ------------------------------------------
python C:\Users\User\Desktop\bgCombine\synthetic_gen_gui.py

REM 3. 程式結束後暫停，以便查看錯誤訊息
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ------------------------------------------
    echo [錯誤] 程式發生錯誤或崩潰 (Code: %ERRORLEVEL%)
    echo 請檢查上方錯誤訊息，確認是否缺少套件或路徑錯誤。
) else (
    echo.
    echo [資訊] 程式已正常結束。
)

exit