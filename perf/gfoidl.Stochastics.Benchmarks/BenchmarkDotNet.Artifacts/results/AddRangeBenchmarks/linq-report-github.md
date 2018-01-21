``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|   Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|--------- |----------:|----------:|----------:|-------:|---------:|
|  Default |  49.40 us | 0.9664 us | 0.8070 us |   1.00 |     0.00 |
| AddRange | 178.76 us | 2.7421 us | 2.5650 us |   3.62 |     0.08 |
