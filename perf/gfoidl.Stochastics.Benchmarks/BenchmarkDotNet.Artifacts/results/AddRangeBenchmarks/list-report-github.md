``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=4
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|   Method |     Mean |     Error |    StdDev | Scaled | ScaledSD |
|--------- |---------:|----------:|----------:|-------:|---------:|
|  Default | 25.96 us | 0.5068 us | 0.7890 us |   1.00 |     0.00 |
| AddRange | 25.27 us | 0.4848 us | 0.6797 us |   0.97 |     0.04 |
