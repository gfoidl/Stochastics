``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|       Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|------------- |----------:|----------:|----------:|-------:|---------:|
|         Erf1 | 106.94 ns | 0.6356 ns | 0.5945 ns |   1.00 |     0.00 |
|         Erf2 |  12.88 ns | 0.2967 ns | 0.4706 ns |   0.12 |     0.00 |
|  Erf2Caching |  43.57 ns | 0.9093 ns | 1.7519 ns |   0.41 |     0.02 |
| Erf2Caching1 |  21.12 ns | 0.4624 ns | 0.6482 ns |   0.20 |     0.01 |
