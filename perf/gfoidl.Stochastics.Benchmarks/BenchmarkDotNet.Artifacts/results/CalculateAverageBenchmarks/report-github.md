``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                 Method |       N |          Mean |         Error |        StdDev | Scaled | ScaledSD |
|----------------------- |-------- |--------------:|--------------:|--------------:|-------:|---------:|
|             **UnsafeSimd** |     **100** |      **97.70 ns** |      **1.530 ns** |      **1.356 ns** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |     100 |  26,473.80 ns |    929.853 ns |  2,741.692 ns | 271.01 |    28.16 |
|             **UnsafeSimd** |    **1000** |     **778.18 ns** |      **2.729 ns** |      **2.553 ns** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |    1000 |  21,562.80 ns |    537.789 ns |  1,551.643 ns |  27.71 |     1.99 |
|             **UnsafeSimd** |   **10000** |   **7,589.01 ns** |     **21.531 ns** |     **20.140 ns** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |   10000 |  24,853.68 ns |    490.125 ns |    920.573 ns |   3.27 |     0.12 |
|             **UnsafeSimd** |  **100000** |  **76,781.94 ns** |    **525.347 ns** |    **465.706 ns** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd |  100000 |  49,217.29 ns |    969.316 ns |  1,116.265 ns |   0.64 |     0.01 |
|             **UnsafeSimd** | **1000000** | **901,205.68 ns** | **29,655.312 ns** | **86,973.897 ns** |   **1.00** |     **0.00** |
| ParallelizedUnsafeSimd | 1000000 | 255,093.38 ns |  3,361.747 ns |  2,980.101 ns |   0.29 |     0.03 |
