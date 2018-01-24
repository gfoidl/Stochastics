``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|        Method |      Mean |    Error |   StdDev | Scaled | ScaledSD |
|-------------- |----------:|---------:|---------:|-------:|---------:|
|          Erf1 | 251.88 ns | 4.964 ns | 4.875 ns |   1.00 |     0.00 |
|          Erf2 | 147.97 ns | 2.966 ns | 5.715 ns |   0.59 |     0.02 |
|        ErfCpp |  81.19 ns | 1.882 ns | 3.844 ns |   0.32 |     0.02 |
| ErfCpp_Vector |  81.95 ns | 1.501 ns | 1.331 ns |   0.33 |     0.01 |
