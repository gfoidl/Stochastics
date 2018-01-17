``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|         Method |      N |         Mean |         Error |        StdDev | Scaled | ScaledSD |
|--------------- |------- |-------------:|--------------:|--------------:|-------:|---------:|
|   **EachSeparate** |    **100** |     **415.8 ns** |     **4.1748 ns** |     **3.9052 ns** |   **1.00** |     **0.00** |
| SimdSequential |    100 |     203.1 ns |     0.6954 ns |     0.6164 ns |   0.49 |     0.00 |
|   SimdParallel |    100 |  19,301.0 ns |   274.3979 ns |   243.2466 ns |  46.43 |     0.70 |
|   **EachSeparate** |   **1000** |   **2,387.2 ns** |    **10.3893 ns** |     **9.7181 ns** |   **1.00** |     **0.00** |
| SimdSequential |   1000 |   1,672.4 ns |     1.7651 ns |     1.5647 ns |   0.70 |     0.00 |
|   SimdParallel |   1000 |  18,579.0 ns |   294.2237 ns |   260.8217 ns |   7.78 |     0.11 |
|   **EachSeparate** |  **10000** |  **21,963.3 ns** |    **79.6382 ns** |    **70.5971 ns** |   **1.00** |     **0.00** |
| SimdSequential |  10000 |  16,307.7 ns |    38.3462 ns |    35.8691 ns |   0.74 |     0.00 |
|   SimdParallel |  10000 |  21,677.5 ns |   297.9120 ns |   264.0912 ns |   0.99 |     0.01 |
|   **EachSeparate** |  **50000** | **150,579.4 ns** | **6,245.7260 ns** | **6,134.1374 ns** |   **1.00** |     **0.00** |
| SimdSequential |  50000 |  83,006.0 ns |   129.9797 ns |   121.5831 ns |   0.55 |     0.02 |
|   SimdParallel |  50000 |  57,336.5 ns | 1,097.9892 ns | 1,348.4294 ns |   0.38 |     0.02 |
|   **EachSeparate** | **100000** | **227,887.2 ns** | **1,879.8884 ns** | **1,569.7918 ns** |   **1.00** |     **0.00** |
| SimdSequential | 100000 | 166,408.4 ns |   690.2975 ns |   645.7047 ns |   0.73 |     0.01 |
|   SimdParallel | 100000 |  85,729.1 ns |   880.7130 ns |   823.8194 ns |   0.38 |     0.00 |
