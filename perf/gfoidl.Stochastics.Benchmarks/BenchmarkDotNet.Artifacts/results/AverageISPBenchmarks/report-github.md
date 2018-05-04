``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.371)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742191 Hz, Resolution=364.6719 ns, Timer=TSC
.NET Core SDK=2.1.300-preview3-008618
  [Host]     : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT


```
| Method |     Mean |     Error |    StdDev | Scaled |
|------- |---------:|----------:|----------:|-------:|
|   Base | 3.135 us | 0.0275 us | 0.0257 us |   1.00 |
|    ISP | 1.259 us | 0.0100 us | 0.0088 us |   0.40 |
|   ISP1 | 1.244 us | 0.0075 us | 0.0067 us |   0.40 |
