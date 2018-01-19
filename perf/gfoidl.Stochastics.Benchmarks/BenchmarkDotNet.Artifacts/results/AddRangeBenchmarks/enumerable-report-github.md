``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|        Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|-------------- |----------:|----------:|----------:|-------:|---------:|
|       Default |  47.16 us | 0.8892 us | 1.2465 us |   1.00 |     0.00 |
|      AddRange | 187.82 us | 3.7229 us | 5.6852 us |   3.99 |     0.16 |
| AddRangeArray |  11.60 us | 0.0401 us | 0.0313 us |   0.25 |     0.01 |
