``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.371)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742187 Hz, Resolution=364.6724 ns, Timer=TSC
.NET Core SDK=2.1.300-preview3-008618
  [Host]     : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT


```
| Method |     Mean |     Error |    StdDev | Scaled | ScaledSD |
|------- |---------:|----------:|----------:|-------:|---------:|
|   Base | 3.358 us | 0.0724 us | 0.0916 us |   1.00 |     0.00 |
|    ISP | 2.649 us | 0.0439 us | 0.0410 us |   0.79 |     0.02 |
|   ISP1 | 2.631 us | 0.0601 us | 0.1021 us |   0.78 |     0.04 |
