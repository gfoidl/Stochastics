``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|           Method |       N |       Mean |     Error |     StdDev | Scaled | ScaledSD |
|----------------- |-------- |-----------:|----------:|-----------:|-------:|---------:|
|             **Simd** | **1000000** |   **623.6 us** |  **3.210 us** |   **2.681 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 1000000 |   658.0 us | 12.951 us |  20.164 us |   1.06 |     0.03 |
|             **Simd** | **1500000** |   **992.2 us** | **18.138 us** |  **20.161 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 1500000 |   953.6 us | 12.293 us |  11.499 us |   0.96 |     0.02 |
|             **Simd** | **2000000** | **1,676.3 us** | **42.982 us** | **124.013 us** |   **1.00** |     **0.00** |
| ParallelizedSimd | 2000000 | 1,304.4 us | 21.566 us |  19.118 us |   0.78 |     0.06 |
