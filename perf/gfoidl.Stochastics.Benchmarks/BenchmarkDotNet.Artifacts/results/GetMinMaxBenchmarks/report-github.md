``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|           Method |       N |       Mean |    Error |    StdDev |   Median | Scaled | ScaledSD |
|----------------- |-------- |-----------:|---------:|----------:|---------:|-------:|---------:|
|             **Simd** | **1750000** |   **643.2 us** | **14.80 us** |  **13.12 us** | **641.8 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 1750000 |   641.1 us | 12.29 us |  12.62 us | 637.4 us |   1.00 |     0.03 |
|             **Simd** | **2000000** | **1,025.2 us** | **76.40 us** | **222.86 us** | **931.0 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 2000000 |   726.4 us | 12.46 us |  11.65 us | 726.8 us |   0.74 |     0.14 |
