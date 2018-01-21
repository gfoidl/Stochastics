``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|        Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|-------------- |----------:|----------:|----------:|-------:|---------:|
|       Default | 300.69 us | 5.4612 us | 5.1084 us |   1.00 |     0.00 |
|      AddRange | 281.22 us | 2.5388 us | 2.3748 us |   0.94 |     0.02 |
| AddRangeArray |  31.07 us | 0.2158 us | 0.2018 us |   0.10 |     0.00 |
