``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                       Method |      N |          Mean |         Error |        StdDev | Scaled | ScaledSD |
|----------------------------- |------- |--------------:|--------------:|--------------:|-------:|---------:|
|           **CombinedSequential** |    **100** |      **87.89 ns** |     **0.6666 ns** |     **0.5910 ns** |   **1.00** |     **0.00** |
| CombinedSequentialInclMinMax |    100 |     216.81 ns |     2.3108 ns |     2.0485 ns |   2.47 |     0.03 |
|             CombinedParallel |    100 |  18,870.84 ns |   188.8870 ns |   167.4434 ns | 214.72 |     2.30 |
|   CombinedParallelInclMinMax |    100 |  17,960.66 ns |   183.3128 ns |   162.5020 ns | 204.36 |     2.22 |
|           **CombinedSequential** |   **1000** |     **584.67 ns** |    **10.2208 ns** |     **9.5605 ns** |   **1.00** |     **0.00** |
| CombinedSequentialInclMinMax |   1000 |   1,684.07 ns |     5.9215 ns |     5.2492 ns |   2.88 |     0.05 |
|             CombinedParallel |   1000 |  17,954.26 ns |    96.2580 ns |    90.0398 ns |  30.72 |     0.50 |
|   CombinedParallelInclMinMax |   1000 |  20,347.10 ns |   260.4267 ns |   243.6032 ns |  34.81 |     0.68 |
|           **CombinedSequential** |  **10000** |   **5,494.06 ns** |    **25.7565 ns** |    **21.5079 ns** |   **1.00** |     **0.00** |
| CombinedSequentialInclMinMax |  10000 |  16,488.86 ns |   115.8293 ns |   108.3468 ns |   3.00 |     0.02 |
|             CombinedParallel |  10000 |  23,785.38 ns |   368.6565 ns |   307.8449 ns |   4.33 |     0.06 |
|   CombinedParallelInclMinMax |  10000 |  22,696.44 ns |   226.4733 ns |   211.8432 ns |   4.13 |     0.04 |
|           **CombinedSequential** | **100000** |  **57,926.12 ns** |   **313.0880 ns** |   **261.4427 ns** |   **1.00** |     **0.00** |
| CombinedSequentialInclMinMax | 100000 | 168,413.60 ns | 1,709.6884 ns | 1,515.5941 ns |   2.91 |     0.03 |
|             CombinedParallel | 100000 |  52,216.74 ns |   309.8169 ns |   289.8029 ns |   0.90 |     0.01 |
|   CombinedParallelInclMinMax | 100000 |  94,776.48 ns |   803.8134 ns |   751.8874 ns |   1.64 |     0.01 |
