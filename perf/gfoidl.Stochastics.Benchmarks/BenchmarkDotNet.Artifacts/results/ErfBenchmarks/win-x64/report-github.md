``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 10 Redstone 3 [1709, Fall Creators Update] (10.0.16299.371)
Processor=Intel Core i7-7700HQ CPU 2.80GHz (Kaby Lake), ProcessorCount=8
Frequency=2742191 Hz, Resolution=364.6719 ns, Timer=TSC
.NET Core SDK=2.1.300-preview3-008618
  [Host]     : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview3-26411-06 (Framework 4.6.26411.07), 64bit RyuJIT


```
|       Method |      Mean |     Error |    StdDev | Scaled |
|------------- |----------:|----------:|----------:|-------:|
|      NetCore | 135.07 us | 1.4571 us | 1.3629 us |   1.00 |
| NativeStdLib |  75.44 us | 0.7772 us | 0.6490 us |   0.56 |
|       Native | 117.94 us | 1.3194 us | 1.2342 us |   0.87 |
