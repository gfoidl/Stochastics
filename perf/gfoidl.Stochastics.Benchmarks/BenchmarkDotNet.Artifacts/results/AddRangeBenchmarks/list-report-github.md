``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|        Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|-------------- |----------:|----------:|----------:|-------:|---------:|
|       Default |  23.32 us | 0.4539 us | 0.4246 us |   1.00 |     0.00 |
|      AddRange | 189.78 us | 4.3587 us | 3.8639 us |   8.14 |     0.21 |
| AddRangeArray |  11.59 us | 0.0520 us | 0.0461 us |   0.50 |     0.01 |
