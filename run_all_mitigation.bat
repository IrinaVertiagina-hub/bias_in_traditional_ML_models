@echo off
echo Running all mitigation experiments...

:: ─── ADULT ───
echo.
echo === ADULT ===

py mitigation.py --dataset adult --model lr --C 0.01
py mitigation.py --dataset adult --model lr --C 0.1
py mitigation.py --dataset adult --model lr --C 1.0
py mitigation.py --dataset adult --model lr --C 10.0
py mitigation.py --dataset adult --model lr --C 100.0

py mitigation.py --dataset adult --model dt --max_depth 3
py mitigation.py --dataset adult --model dt --max_depth 7
py mitigation.py --dataset adult --model dt --max_depth 10
py mitigation.py --dataset adult --model dt --max_depth 15

py mitigation.py --dataset adult --model svm --C 0.1 --kernel linear
py mitigation.py --dataset adult --model svm --C 1.0 --kernel linear
py mitigation.py --dataset adult --model svm --C 10.0 --kernel linear
py mitigation.py --dataset adult --model svm --C 0.1 --kernel rbf
py mitigation.py --dataset adult --model svm --C 1.0 --kernel rbf
py mitigation.py --dataset adult --model svm --C 10.0 --kernel rbf

py mitigation.py --dataset adult --model knn --n_neighbors 3
py mitigation.py --dataset adult --model knn --n_neighbors 7
py mitigation.py --dataset adult --model knn --n_neighbors 15
py mitigation.py --dataset adult --model knn --n_neighbors 25

:: ─── COMPAS ───
echo.
echo === COMPAS ===

py mitigation.py --dataset compas --model lr --C 0.01
py mitigation.py --dataset compas --model lr --C 0.1
py mitigation.py --dataset compas --model lr --C 1.0
py mitigation.py --dataset compas --model lr --C 10.0
py mitigation.py --dataset compas --model lr --C 100.0

py mitigation.py --dataset compas --model dt --max_depth 3
py mitigation.py --dataset compas --model dt --max_depth 7
py mitigation.py --dataset compas --model dt --max_depth 10
py mitigation.py --dataset compas --model dt --max_depth 15

py mitigation.py --dataset compas --model svm --C 0.1 --kernel linear
py mitigation.py --dataset compas --model svm --C 1.0 --kernel linear
py mitigation.py --dataset compas --model svm --C 10.0 --kernel linear
py mitigation.py --dataset compas --model svm --C 0.1 --kernel rbf
py mitigation.py --dataset compas --model svm --C 1.0 --kernel rbf
py mitigation.py --dataset compas --model svm --C 10.0 --kernel rbf

py mitigation.py --dataset compas --model knn --n_neighbors 3
py mitigation.py --dataset compas --model knn --n_neighbors 7
py mitigation.py --dataset compas --model knn --n_neighbors 15
py mitigation.py --dataset compas --model knn --n_neighbors 25

:: ─── GERMAN ───
echo.
echo === GERMAN ===

py mitigation.py --dataset german --model lr --C 0.01
py mitigation.py --dataset german --model lr --C 0.1
py mitigation.py --dataset german --model lr --C 1.0
py mitigation.py --dataset german --model lr --C 10.0
py mitigation.py --dataset german --model lr --C 100.0

py mitigation.py --dataset german --model dt --max_depth 3
py mitigation.py --dataset german --model dt --max_depth 7
py mitigation.py --dataset german --model dt --max_depth 10
py mitigation.py --dataset german --model dt --max_depth 15

py mitigation.py --dataset german --model svm --C 0.1 --kernel linear
py mitigation.py --dataset german --model svm --C 1.0 --kernel linear
py mitigation.py --dataset german --model svm --C 10.0 --kernel linear
py mitigation.py --dataset german --model svm --C 0.1 --kernel rbf
py mitigation.py --dataset german --model svm --C 1.0 --kernel rbf
py mitigation.py --dataset german --model svm --C 10.0 --kernel rbf

py mitigation.py --dataset german --model knn --n_neighbors 3
py mitigation.py --dataset german --model knn --n_neighbors 7
py mitigation.py --dataset german --model knn --n_neighbors 15
py mitigation.py --dataset german --model knn --n_neighbors 25

echo.
echo All mitigation experiments done. Results saved to results/