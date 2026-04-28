@echo off
echo Running all experiments...

:: ─── ADULT ───
echo.
echo === ADULT ===

py models.py --dataset adult --model lr --C 0.01
py models.py --dataset adult --model lr --C 0.1
py models.py --dataset adult --model lr --C 1.0
py models.py --dataset adult --model lr --C 10.0
py models.py --dataset adult --model lr --C 100.0

py models.py --dataset adult --model dt --max_depth 3
py models.py --dataset adult --model dt --max_depth 7
py models.py --dataset adult --model dt --max_depth 10
py models.py --dataset adult --model dt --max_depth 15

py models.py --dataset adult --model svm --C 0.1 --kernel linear
py models.py --dataset adult --model svm --C 1.0 --kernel linear
py models.py --dataset adult --model svm --C 10.0 --kernel linear
py models.py --dataset adult --model svm --C 0.1 --kernel rbf
py models.py --dataset adult --model svm --C 1.0 --kernel rbf
py models.py --dataset adult --model svm --C 10.0 --kernel rbf

py models.py --dataset adult --model knn --n_neighbors 3
py models.py --dataset adult --model knn --n_neighbors 7
py models.py --dataset adult --model knn --n_neighbors 15
py models.py --dataset adult --model knn --n_neighbors 25

:: ─── COMPAS ───
echo.
echo === COMPAS ===

py models.py --dataset compas --model lr --C 0.01
py models.py --dataset compas --model lr --C 0.1
py models.py --dataset compas --model lr --C 1.0
py models.py --dataset compas --model lr --C 10.0
py models.py --dataset compas --model lr --C 100.0

py models.py --dataset compas --model dt --max_depth 3
py models.py --dataset compas --model dt --max_depth 7
py models.py --dataset compas --model dt --max_depth 10
py models.py --dataset compas --model dt --max_depth 15

py models.py --dataset compas --model svm --C 0.1 --kernel linear
py models.py --dataset compas --model svm --C 1.0 --kernel linear
py models.py --dataset compas --model svm --C 10.0 --kernel linear
py models.py --dataset compas --model svm --C 0.1 --kernel rbf
py models.py --dataset compas --model svm --C 1.0 --kernel rbf
py models.py --dataset compas --model svm --C 10.0 --kernel rbf

py models.py --dataset compas --model knn --n_neighbors 3
py models.py --dataset compas --model knn --n_neighbors 7
py models.py --dataset compas --model knn --n_neighbors 15
py models.py --dataset compas --model knn --n_neighbors 25

:: ─── GERMAN ───
echo.
echo === GERMAN ===

py models.py --dataset german --model lr --C 0.01
py models.py --dataset german --model lr --C 0.1
py models.py --dataset german --model lr --C 1.0
py models.py --dataset german --model lr --C 10.0
py models.py --dataset german --model lr --C 100.0

py models.py --dataset german --model dt --max_depth 3
py models.py --dataset german --model dt --max_depth 7
py models.py --dataset german --model dt --max_depth 10
py models.py --dataset german --model dt --max_depth 15

py models.py --dataset german --model svm --C 0.1 --kernel linear
py models.py --dataset german --model svm --C 1.0 --kernel linear
py models.py --dataset german --model svm --C 10.0 --kernel linear
py models.py --dataset german --model svm --C 0.1 --kernel rbf
py models.py --dataset german --model svm --C 1.0 --kernel rbf
py models.py --dataset german --model svm --C 10.0 --kernel rbf

py models.py --dataset german --model knn --n_neighbors 3
py models.py --dataset german --model knn --n_neighbors 7
py models.py --dataset german --model knn --n_neighbors 15
py models.py --dataset german --model knn --n_neighbors 25

echo.
echo All experiments done. Results saved to results/