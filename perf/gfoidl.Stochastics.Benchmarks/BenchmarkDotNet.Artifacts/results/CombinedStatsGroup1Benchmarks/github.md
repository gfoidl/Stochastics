``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|         Method |      N |            Mean |          Error |         StdDev | Scaled | ScaledSD |
|--------------- |------- |----------------:|---------------:|---------------:|-------:|---------:|
|   **EachSeparate** |    **100** |     **1,938.49 ns** |     **37.8843 ns** |     **40.5357 ns** |   **1.00** |     **0.00** |
| SimdSequential |    100 |        93.58 ns |      0.8093 ns |      0.7570 ns |   0.05 |     0.00 |
|   SimdParallel |    100 |    18,570.79 ns |    163.6505 ns |    153.0787 ns |   9.58 |     0.21 |
|   **EachSeparate** |   **1000** |    **17,418.82 ns** |    **348.6196 ns** |    **415.0068 ns** |   **1.00** |     **0.00** |
| SimdSequential |   1000 |       582.92 ns |      3.7374 ns |      3.4960 ns |   0.03 |     0.00 |
|   SimdParallel |   1000 |    17,814.86 ns |    246.9849 ns |    231.0298 ns |   1.02 |     0.03 |
|   **EachSeparate** |  **10000** |   **167,245.70 ns** |  **2,639.7801 ns** |  **2,340.0960 ns** |   **1.00** |     **0.00** |
| SimdSequential |  10000 |     5,472.29 ns |      8.4690 ns |      7.9219 ns |   0.03 |     0.00 |
|   SimdParallel |  10000 |    24,553.67 ns |    484.8168 ns |    453.4979 ns |   0.15 |     0.00 |
|   **EachSeparate** |  **50000** |   **839,303.99 ns** | **14,030.7323 ns** | **13,124.3544 ns** |   **1.00** |     **0.00** |
| SimdSequential |  50000 |    28,359.36 ns |     47.9573 ns |     42.5129 ns |   0.03 |     0.00 |
|   SimdParallel |  50000 |    31,517.43 ns |    343.0945 ns |    320.9308 ns |   0.04 |     0.00 |
|   **EachSeparate** | **100000** | **1,683,470.07 ns** | **14,576.2300 ns** | **12,921.4470 ns** |   **1.00** |     **0.00** |
| SimdSequential | 100000 |    56,845.20 ns |    128.9509 ns |    114.3116 ns |   0.03 |     0.00 |
|   SimdParallel | 100000 |    44,860.82 ns |    292.9488 ns |    274.0244 ns |   0.03 |     0.00 |
