``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|             Method |      N |         Mean |         Error |        StdDev |       Median | Scaled | ScaledSD |
|------------------- |------- |-------------:|--------------:|--------------:|-------------:|-------:|---------:|
|       **EachSeparate** |    **100** |    **223.70 ns** |     **3.2335 ns** |     **2.8664 ns** |    **223.33 ns** |   **1.00** |     **0.00** |
| CombinedSequential |    100 |     88.46 ns |     0.8312 ns |     0.7369 ns |     88.32 ns |   0.40 |     0.01 |
|   CombinedParallel |    100 | 18,522.51 ns |   366.2474 ns |   407.0830 ns | 18,413.27 ns |  82.81 |     2.05 |
|       **EachSeparate** |   **1000** |  **1,213.51 ns** |     **5.5146 ns** |     **4.8886 ns** |  **1,212.55 ns** |   **1.00** |     **0.00** |
| CombinedSequential |   1000 |    581.02 ns |     3.8832 ns |     3.4424 ns |    579.54 ns |   0.48 |     0.00 |
|   CombinedParallel |   1000 | 19,462.43 ns |   352.1782 ns |   345.8861 ns | 19,374.61 ns |  16.04 |     0.28 |
|       **EachSeparate** |  **10000** | **11,069.60 ns** |    **44.6306 ns** |    **39.5639 ns** | **11,073.71 ns** |   **1.00** |     **0.00** |
| CombinedSequential |  10000 |  5,498.95 ns |    29.2967 ns |    27.4042 ns |  5,501.52 ns |   0.50 |     0.00 |
|   CombinedParallel |  10000 | 23,589.70 ns |   414.6571 ns |   346.2574 ns | 23,631.85 ns |   2.13 |     0.03 |
|       **EachSeparate** |  **50000** | **64,718.66 ns** | **1,274.4054 ns** | **1,416.4981 ns** | **64,113.00 ns** |   **1.00** |     **0.00** |
| CombinedSequential |  50000 | 29,335.35 ns |   299.4284 ns |   280.0855 ns | 29,346.81 ns |   0.45 |     0.01 |
|   CombinedParallel |  50000 | 35,333.19 ns |   681.1540 ns |   637.1518 ns | 35,268.31 ns |   0.55 |     0.01 |
|       **EachSeparate** | **100000** | **92,605.60 ns** | **1,939.4370 ns** | **4,090.9308 ns** | **90,826.05 ns** |   **1.00** |     **0.00** |
| CombinedSequential | 100000 | 58,233.29 ns |   464.6050 ns |   434.5917 ns | 58,206.33 ns |   0.63 |     0.03 |
|   CombinedParallel | 100000 | 53,609.49 ns | 1,057.6188 ns |   989.2972 ns | 53,147.28 ns |   0.58 |     0.03 |
