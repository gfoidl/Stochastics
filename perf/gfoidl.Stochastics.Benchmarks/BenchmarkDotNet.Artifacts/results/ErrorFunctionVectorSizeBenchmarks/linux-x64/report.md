``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.4
  [Host]     : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.5 (Framework 4.6.0.0), 64bit RyuJIT


```
|           Method |     Mean |     Error |    StdDev | Scaled |
|----------------- |---------:|----------:|----------:|-------:|
|          NetBase | 38.71 ms | 0.0823 ms | 0.0730 ms |   1.00 |
|  ErfCpp_Vector_8 | 33.28 ms | 0.1596 ms | 0.1493 ms |   0.86 |
| ErfCpp_Vector_10 | 33.36 ms | 0.1355 ms | 0.1267 ms |   0.86 |
| ErfCpp_Vector_16 | 33.06 ms | 0.0656 ms | 0.0614 ms |   0.85 |
| ErfCpp_Vector_32 | 32.85 ms | 0.0682 ms | 0.0604 ms |   0.85 |
| ErfCpp_Vector_64 | 32.72 ms | 0.0544 ms | 0.0509 ms |   0.85 |
