``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                 Method |      N |     Mean |     Error |    StdDev | Scaled | ScaledSD |
|----------------------- |------- |---------:|----------:|----------:|-------:|---------:|
|             **UnsafeSimd** |  **50000** | **37.87 us** | **0.1356 us** | **0.1269 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |  50000 | 35.32 us | 0.7003 us | 1.1892 us |   0.93 |     0.03 |
|             **UnsafeSimd** |  **75000** | **57.30 us** | **0.3970 us** | **0.3713 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |  75000 | 45.28 us | 0.9020 us | 1.8629 us |   0.79 |     0.03 |
|             **UnsafeSimd** | **100000** | **76.39 us** | **0.4196 us** | **0.3925 us** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd | 100000 | 50.09 us | 1.1897 us | 3.5079 us |   0.66 |     0.05 |
