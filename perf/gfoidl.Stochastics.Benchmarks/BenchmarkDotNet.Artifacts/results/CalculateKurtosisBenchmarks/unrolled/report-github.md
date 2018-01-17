``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|             Method |       N |           Mean |          Error |         StdDev | Scaled |
|------------------- |-------- |---------------:|---------------:|---------------:|-------:|
|         **UnsafeSimd** |     **100** |       **286.6 ns** |      **1.4332 ns** |      **1.3406 ns** |   **1.00** |
| UnsafeSimdUnrolled |     100 |       277.7 ns |      0.8934 ns |      0.8357 ns |   0.97 |
|         **UnsafeSimd** |    **1000** |     **2,667.5 ns** |      **8.7674 ns** |      **7.7721 ns** |   **1.00** |
| UnsafeSimdUnrolled |    1000 |     2,658.7 ns |     13.1307 ns |     12.2825 ns |   1.00 |
|         **UnsafeSimd** |   **10000** |    **26,410.1 ns** |     **84.3692 ns** |     **74.7911 ns** |   **1.00** |
| UnsafeSimdUnrolled |   10000 |    26,488.3 ns |    113.0171 ns |    100.1867 ns |   1.00 |
|         **UnsafeSimd** |  **100000** |   **270,792.0 ns** |  **4,111.9347 ns** |  **3,846.3059 ns** |   **1.00** |
| UnsafeSimdUnrolled |  100000 |   267,459.2 ns |  1,177.1100 ns |  1,101.0693 ns |   0.99 |
|         **UnsafeSimd** | **1000000** | **2,803,335.9 ns** | **31,686.7146 ns** | **29,639.7696 ns** |   **1.00** |
| UnsafeSimdUnrolled | 1000000 | 2,782,048.5 ns | 15,356.2713 ns | 13,612.9332 ns |   0.99 |
