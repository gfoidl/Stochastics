``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.125)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742191 Hz, Resolution=364.6719 ns, Timer=TSC
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT


```
|       Method |     Mean |     Error |    StdDev | Scaled | ScaledSD |
|------------- |---------:|----------:|----------:|-------:|---------:|
|         Erf1 | 13.38 ns | 0.2774 ns | 0.3084 ns |   1.00 |     0.00 |
|         Erf2 | 12.92 ns | 0.2918 ns | 0.4457 ns |   0.97 |     0.04 |
|  Erf2Caching | 26.48 ns | 0.5341 ns | 0.4996 ns |   1.98 |     0.06 |
| Erf2Caching1 | 15.55 ns | 0.2163 ns | 0.2023 ns |   1.16 |     0.03 |
