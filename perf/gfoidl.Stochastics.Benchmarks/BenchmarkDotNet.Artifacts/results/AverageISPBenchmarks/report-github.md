``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.371)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742187 Hz, Resolution=364.6724 ns, Timer=TSC
.NET Core SDK=2.1.300-preview3-008618
  [Host]     : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT


```
| Method |     Mean |     Error |    StdDev | Scaled |
|------- |---------:|----------:|----------:|-------:|
|   Base | 3.224 us | 0.0395 us | 0.0330 us |   1.00 |
|    ISP | 2.622 us | 0.0279 us | 0.0261 us |   0.81 |
|   ISP1 | 1.235 us | 0.0090 us | 0.0085 us |   0.38 |
