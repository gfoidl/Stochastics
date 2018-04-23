``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|           Method |    N |     Mean |     Error |    StdDev | Scaled |
|----------------- |----- |---------:|----------:|----------:|-------:|
|             **Simd** | **1500** | **633.7 us** | **0.9077 us** | **0.8490 us** |   **1.00** |
| ParallelizedSimd | 1500 | 638.2 us | 3.1703 us | 2.8104 us |   1.01 |
|             **Simd** | **1750** | **862.1 us** | **1.5114 us** | **1.4138 us** |   **1.00** |
| ParallelizedSimd | 1750 | 746.7 us | 1.8980 us | 1.6825 us |   0.87 |
