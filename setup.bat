@echo off
REM One-time setup on Windows. Requires Python 3.11 or 3.12 (64-bit) on PATH.
echo Creating virtual environment...
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat

echo Installing PyTorch (CPU build)...
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error

echo Installing project dependencies...
pip install -r requirements.txt || goto :error

echo Creating demo workspace...
python scripts\make_demo_corpus.py demo_workspace || goto :error

echo.
echo Setup complete.
echo Next: install Ollama from https://ollama.com, then run:  ollama pull qwen3:1.7b
echo Then start the app with:  run.bat
goto :eof

:error
echo.
echo Setup failed. See the message above.
exit /b 1
