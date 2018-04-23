``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                     Method |      N |            Mean |          Error |        StdDev | Scaled | ScaledSD |
|--------------------------- |------- |----------------:|---------------:|--------------:|-------:|---------:|
|         **UnsafeParallelSimd** |    **100** |        **27.07 us** |      **0.7299 us** |      **1.278 us** |   **1.00** |     **0.00** |
| UnsafeParallelSimdUnrolled |    100 |        26.19 us |      0.5399 us |      1.523 us |   0.97 |     0.07 |
|         **UnsafeParallelSimd** |   **1000** |       **236.67 us** |      **3.2579 us** |      **2.888 us** |   **1.00** |     **0.00** |
| UnsafeParallelSimdUnrolled |   1000 |       238.99 us |      3.4127 us |      3.192 us |   1.01 |     0.02 |
|         **UnsafeParallelSimd** |  **10000** |    **19,100.60 us** |    **133.7510 us** |    **125.111 us** |   **1.00** |     **0.00** |
| UnsafeParallelSimdUnrolled |  10000 |    19,061.65 us |    137.3330 us |    128.461 us |   1.00 |     0.01 |
|         **UnsafeParallelSimd** | **100000** | **2,008,763.74 us** | **13,942.6412 us** | **13,041.954 us** |   **1.00** |     **0.00** |
| UnsafeParallelSimdUnrolled | 100000 | 2,014,718.65 us | 18,308.9585 us | 17,126.209 us |   1.00 |     0.01 |
