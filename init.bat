cd /d %~dp0
mkdir .\data
mkdir .\output
pause
cd lib\utils
python setup_windows.py build_ext --inplace
pause
cd ..\nms
python setup_windows.py build_ext --inplace
python setup_windows_cuda.py build_ext --inplace
pause
cd ..\pycocotools
python setup_windows.py build_ext --inplace

cd ..\..
pause
