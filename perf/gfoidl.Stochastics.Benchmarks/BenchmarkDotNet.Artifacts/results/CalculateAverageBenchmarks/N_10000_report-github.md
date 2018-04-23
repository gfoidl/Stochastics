``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|                 Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|----------------------- |----------:|----------:|----------:|-------:|---------:|
|                   Linq | 74.279 us | 1.5565 us | 2.9235 us |   1.00 |     0.00 |
|                  PLinq | 61.311 us | 1.2145 us | 2.5081 us |   0.83 |     0.05 |
|             UnsafeSimd |  7.599 us | 0.0506 us | 0.0473 us |   0.10 |     0.00 |
| ParallelizedUnsafeSimd | 25.995 us | 0.4998 us | 0.5556 us |   0.35 |     0.02 |
