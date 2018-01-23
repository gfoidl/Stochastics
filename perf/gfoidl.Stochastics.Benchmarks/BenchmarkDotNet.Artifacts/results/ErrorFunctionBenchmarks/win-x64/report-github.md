``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.125)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742191 Hz, Resolution=364.6719 ns, Timer=TSC
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT


```
|        Method |     Mean |    Error |   StdDev | Scaled | ScaledSD |
|-------------- |---------:|---------:|---------:|-------:|---------:|
|          Erf1 | 151.0 ns | 3.031 ns | 3.491 ns |   1.00 |     0.00 |
|          Erf2 | 140.4 ns | 2.583 ns | 4.386 ns |   0.93 |     0.04 |
|        ErfCpp | 146.3 ns | 2.345 ns | 2.194 ns |   0.97 |     0.03 |
| ErfCpp_Vector | 109.6 ns | 2.147 ns | 2.556 ns |   0.73 |     0.02 |
