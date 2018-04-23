``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|           Method |       N |     Mean |     Error |    StdDev | Scaled | ScaledSD |
|----------------- |-------- |---------:|----------:|----------:|-------:|---------:|
|             **Simd** | **1250000** | **462.0 us** |  **9.056 us** |  **8.028 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 1250000 | 479.9 us |  9.523 us | 12.043 us |   1.04 |     0.03 |
|             **Simd** | **1500000** | **606.0 us** | **12.236 us** | **30.697 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 1500000 | 571.7 us |  7.548 us |  6.303 us |   0.95 |     0.05 |
