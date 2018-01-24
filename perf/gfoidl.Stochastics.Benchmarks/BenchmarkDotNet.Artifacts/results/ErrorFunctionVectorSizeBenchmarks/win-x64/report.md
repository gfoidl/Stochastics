``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.125)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742191 Hz, Resolution=364.6719 ns, Timer=TSC
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.26020.03), 64bit RyuJIT


```
|           Method |      Mean |     Error |    StdDev | Scaled |
|----------------- |----------:|----------:|----------:|-------:|
|          NetBase | 14.350 ms | 0.2333 ms | 0.2182 ms |   1.00 |
|  ErfCpp_Vector_8 | 11.502 ms | 0.1048 ms | 0.0875 ms |   0.80 |
| ErfCpp_Vector_10 | 10.726 ms | 0.1791 ms | 0.1495 ms |   0.75 |
| ErfCpp_Vector_16 | 10.027 ms | 0.0672 ms | 0.0595 ms |   0.70 |
| ErfCpp_Vector_32 |  9.889 ms | 0.1452 ms | 0.1358 ms |   0.69 |
| ErfCpp_Vector_64 |  9.832 ms | 0.1608 ms | 0.1504 ms |   0.69 |
