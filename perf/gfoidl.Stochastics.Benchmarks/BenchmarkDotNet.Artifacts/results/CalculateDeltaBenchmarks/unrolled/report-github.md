``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                         Method |       N |      Mean |     Error |     StdDev |    Median | Scaled | ScaledSD |
|------------------------------- |-------- |----------:|----------:|-----------:|----------:|-------:|---------:|
|         **ParallelizedUnsafeSimd** |     **100** |  **21.89 us** | **0.7062 us** |  **2.0712 us** |  **21.58 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimdUnrolled |     100 |  21.07 us | 0.4978 us |  1.4601 us |  21.09 us |   0.97 |     0.11 |
|         **ParallelizedUnsafeSimd** |    **1000** |  **19.09 us** | **0.3780 us** |  **0.9690 us** |  **18.87 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimdUnrolled |    1000 |  19.11 us | 0.3714 us |  0.5326 us |  19.14 us |   1.00 |     0.06 |
|         **ParallelizedUnsafeSimd** |   **10000** |  **27.21 us** | **0.5433 us** |  **1.4964 us** |  **26.84 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimdUnrolled |   10000 |  26.94 us | 0.5121 us |  0.6289 us |  26.80 us |   0.99 |     0.06 |
|         **ParallelizedUnsafeSimd** |  **100000** |  **72.00 us** | **1.4257 us** |  **3.0074 us** |  **71.44 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimdUnrolled |  100000 |  69.91 us | 1.3516 us |  1.8500 us |  69.84 us |   0.97 |     0.05 |
|         **ParallelizedUnsafeSimd** | **1000000** | **422.23 us** | **4.2912 us** |  **3.8040 us** | **423.28 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimdUnrolled | 1000000 | 419.31 us | 8.1985 us | 14.1420 us | 412.90 us |   0.99 |     0.03 |
