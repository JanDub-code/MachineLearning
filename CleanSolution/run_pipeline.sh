#!/bin/bash
# ========================================================================
# CleanSolution - Automatické spuštění celého pipeline
# ========================================================================

echo ""
echo "========================================================================"
echo "  CleanSolution - Stock Price Prediction Pipeline"
echo "========================================================================"
echo ""

cd scripts

echo "[1/4] FÁZE 2: Stahování fundamentálních dat..."
echo "Odhadovaný čas: 30-45 minut"
echo ""
python 1_download_fundamentals.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHYBA: Skript 1 selhal!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[2/4] FÁZE 3: Trénování AI modelu..."
echo "Odhadovaný čas: 5-10 minut"
echo ""
python 2_train_fundamental_predictor.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHYBA: Skript 2 selhal!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[3/4] FÁZE 4: Doplňování historických dat..."
echo "Odhadovaný čas: 5-10 minut"
echo ""
python 3_complete_historical_data.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHYBA: Skript 3 selhal!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[4/4] FÁZE 5: Trénování predikčního modelu..."
echo "Odhadovaný čas: 5-10 minut"
echo ""
python 4_train_price_predictor.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CHYBA: Skript 4 selhal!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  ✅ PIPELINE KOMPLETNĚ DOKONČEN!"
echo "========================================================================"
echo ""
echo "Výsledky najdete v:"
echo "  • ../models/ (natrénované modely)"
echo "  • ../data/complete/ (kompletní dataset)"
echo "  • ../data/analysis/ (metriky a vizualizace)"
echo ""
echo "Děkujeme za použití CleanSolution!"
echo ""
