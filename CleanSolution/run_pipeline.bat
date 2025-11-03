@echo off
REM ========================================================================
REM CleanSolution - Automatické spuštění celého pipeline
REM ========================================================================

echo.
echo ========================================================================
echo   CleanSolution - Stock Price Prediction Pipeline
echo ========================================================================
echo.

cd scripts

echo [1/4] FAZE 2: Stahování fundamentálních dat...
echo Odhadovaný čas: 30-45 minut
echo.
python 1_download_fundamentals.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ CHYBA: Skript 1 selhal!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo [2/4] FAZE 3: Trénování AI modelu...
echo Odhadovaný čas: 5-10 minut
echo.
python 2_train_fundamental_predictor.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ CHYBA: Skript 2 selhal!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo [3/4] FAZE 4: Doplňování historických dat...
echo Odhadovaný čas: 5-10 minut
echo.
python 3_complete_historical_data.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ CHYBA: Skript 3 selhal!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo [4/4] FAZE 5: Trénování predikčního modelu...
echo Odhadovaný čas: 5-10 minut
echo.
python 4_train_price_predictor.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ CHYBA: Skript 4 selhal!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo   ✅ PIPELINE KOMPLETNĚ DOKONČEN!
echo ========================================================================
echo.
echo Výsledky najdete v:
echo   • ../models/ (natrénované modely)
echo   • ../data/complete/ (kompletní dataset)
echo   • ../data/analysis/ (metriky a vizualizace)
echo.
echo Děkujeme za použití CleanSolution!
echo.

pause
