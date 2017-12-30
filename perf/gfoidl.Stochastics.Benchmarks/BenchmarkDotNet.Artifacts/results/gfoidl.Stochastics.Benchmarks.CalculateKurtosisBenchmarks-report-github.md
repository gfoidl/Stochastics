``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 7 SP1 (6.1.7601.0)
Processor=Intel Core i7-3610QM CPU 2.30GHz (Ivy Bridge), ProcessorCount=8
Frequency=2241064 Hz, Resolution=446.2166 ns, Timer=TSC
.NET Core SDK=2.1.2
  [Host]     : .NET Core 2.0.3 (Framework 4.6.25815.02), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.3 (Framework 4.6.25815.02), 64bit RyuJIT


```
|     Method |      Mean |     Error |    StdDev | Scaled | ScaledSD |
|----------- |----------:|----------:|----------:|-------:|---------:|
| Sequential | 11.240 us | 0.2209 us | 0.3691 us |   1.00 |     0.00 |
| UnsafeSimd |  5.369 us | 0.1042 us | 0.1391 us |   0.48 |     0.02 |
