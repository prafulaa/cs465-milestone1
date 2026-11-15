@echo off
rem Build and run the Windows-native CUDA smoke test.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
cd /d E:\work\cs465\project\gpu
nvcc -arch=native -DBS=32 mm_windows_test.cu -o mm_win32.exe
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo BUILD_OK
mm_win32.exe 1
mm_win32.exe 0
