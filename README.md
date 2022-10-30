# openacc_heat_equation

## Задание
В данной работе нужно было реализовать уравнение теплопроводности в двумерной области на  равномерных сетках с начальной инициализацией в сетки в граничных точках.

Нужно было реализовать программу на GPU с использованием CUDA, а часть подсчета ошибки реализовать с использованием библиотеки cuBLAS. 

## Ход работы

### <ins>Общие части</ins>

Для удобной реализации уравнения теплопроводности с использованием CUDA на языке C++ были написаны RAII-обертки для аллокаторов памяти на девайсе и хосте, и также для стримов. Эти обертки определены в [данном файле](src/cuda_utils/cuda_utils.cuh). [Аналогичные обертки](src/cuda_utils/cublas_utils.cuh) были написаны для cuBLAS.

### <ins>Версия 1</ins>

[Версия 1](src/heat_equation_solver_cuda_naive.cu) была реализована с учетом знаний, полученных в ходе выполнения предыдущих лабораторных работ: ядра вычислений сетки запускались асинхронно относительно друг друга, были использованы аналогичные шаги для вычисления ошибки **err**.

Как изменилась программа на CUDA в отличие от OpenACC версии:
1. Выделение памяти происходило явно через `cudaMallocPitch` для двумерных массивов, чтобы каждая строка двумерного массива была выровнена для эффективного доступа к ней.
2. Инициализация буфферов сетки и буффера для вычисления ошибки происходило асинхронно в 3-х разных стримах;
3. Для вычисления сетки было написано ядро `grid_recompute`, которое запускалось с количеством нитей в блоке 16x16 и с количеством блоков в сетке (grid_size / 16, grid_size / 16)
4. Ошибка также пересчитывалась с помощью cuBLAS

Ниже приведен профиль наивной версии программы.

![img](img/cuda_naive.png)
<center>Профиль "Версии 1"</center>

По профилю программы можно заметить, что запуски функций из cuBLAS происходят в нулевом стриме, из-за чего происходит слишком ранняя синхронизация стрима.

### <ins>Версия 2</ins>

Во второй версии [программы](src/heat_equation_solver_cuda_without_sync.cu) запуск cuBLAS функций происходил на том же стриме, что и запуск ядер вычисления сетки, из-за чего не было лишнего ожидания лаунчинга cuBLAS-ядер. Явная синхронизация с этим потоком происходила только перед проверкой ошибкой **err**, чтобы определить, нужно ли выходить из цикла. 

![img](img/cuda_without_sync.png)
<center>Профиль "Версии 2"</center>

По профилю видно, что синхронизация происходила еще раньше, после копирования индекса **err_idx** максимальной разницы текущей и предыдущей сеток с девайса на хост, чтобы использовать его для вычисления ошибки **err**. Также видно, что происходит одно лишнее копирования `DtoD`, а также лишний `cudaFree`.

### <ins>Версия 3</ins>

Далее появилась идея избавиться от лишних `DtoD`-копирования и `cudaFree`. Для этого память под ошибку **err** и под индекс ошибки **err_idx** была выделена до начала вычисления сетки, что помогло убрать лишние операции с памятью

![image](img/cuda_once_mem_alloc.png)
<center>Профиль программы "Версия 3"</center>

## Бенчмарки
В результаты разные оптимизации программы на CUDA не дали особенный прирост в скорости работы программы, но все равно, вариант на CUDA оказался во всех случаях быстрее варианта на OpenACC

![image](benchmarks_512_blocks_633000_iters.png)
![image](benchmarks_1024_blocks_1000500_iters.png)
![image](benchmarks_1024_blocks_2389500_iters.png)
![image](benchmarks_1536_blocks_1000500_iters.png)

<center>Бенчмарки разных версий программ</center>
