``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=4
.NET Core SDK=2.1.300-preview3-008387
  [Host]     : .NET Core 2.1.0-preview2-26313-01 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26313-01 (Framework 4.6.26310.01), 64bit RyuJIT


```
|       Method |      Mean |     Error |    StdDev |    Median | Scaled | ScaledSD |
|------------- |----------:|----------:|----------:|----------:|-------:|---------:|
|      NetCore | 148.37 us | 2.9419 us | 5.5255 us | 145.74 us |   1.00 |     0.00 |
| NativeStdLib |  66.11 us | 0.5342 us | 0.4997 us |  66.11 us |   0.45 |     0.02 |
|       Native | 116.09 us | 0.2963 us | 0.2475 us | 116.14 us |   0.78 |     0.03 |
