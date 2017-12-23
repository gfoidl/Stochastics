``` ini

BenchmarkDotNet=v0.10.11, OS=Windows 7 SP1 (6.1.7601.0)
Processor=Intel Core i7-3610QM CPU 2.30GHz (Ivy Bridge), ProcessorCount=8
Frequency=2241054 Hz, Resolution=446.2186 ns, Timer=TSC
.NET Core SDK=2.1.2
  [Host]     : .NET Core 2.0.3 (Framework 4.6.25815.02), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.3 (Framework 4.6.25815.02), 64bit RyuJIT


```
|           Method |       N |            Mean |         Error |        StdDev | Scaled | ScaledSD |
|----------------- |-------- |----------------:|--------------:|--------------:|-------:|---------:|
|       **Sequential** |     **100** |       **103.08 ns** |      **2.530 ns** |      **3.108 ns** |   **1.00** |     **0.00** |
|             Simd |     100 |        77.95 ns |      1.381 ns |      1.418 ns |   0.76 |     0.03 |
|     Parallelized |     100 |    14,260.56 ns |    280.124 ns |    572.218 ns | 138.47 |     6.79 |
| ParallelizedSimd |     100 |    15,109.03 ns |    297.945 ns |    566.871 ns | 146.70 |     6.89 |
|       **Sequential** |    **1000** |       **992.65 ns** |     **10.378 ns** |      **8.666 ns** |   **1.00** |     **0.00** |
|             Simd |    1000 |       758.28 ns |      8.265 ns |      7.731 ns |   0.76 |     0.01 |
|     Parallelized |    1000 |    19,512.83 ns |    388.174 ns |    784.130 ns |  19.66 |     0.80 |
| ParallelizedSimd |    1000 |    18,146.27 ns |    413.119 ns |  1,218.091 ns |  18.28 |     1.23 |
|       **Sequential** |   **10000** |    **10,283.10 ns** |    **170.939 ns** |    **159.896 ns** |   **1.00** |     **0.00** |
|             Simd |   10000 |     9,173.94 ns |     72.481 ns |     67.798 ns |   0.89 |     0.01 |
|     Parallelized |   10000 |    29,444.16 ns |    696.735 ns |  2,054.338 ns |   2.86 |     0.20 |
| ParallelizedSimd |   10000 |    27,063.47 ns |    538.435 ns |  1,464.853 ns |   2.63 |     0.15 |
|       **Sequential** |  **100000** |    **99,172.89 ns** |  **1,279.400 ns** |  **1,134.155 ns** |   **1.00** |     **0.00** |
|             Simd |  100000 |    74,260.29 ns |    155.633 ns |    121.508 ns |   0.75 |     0.01 |
|     Parallelized |  100000 |    69,457.64 ns |  1,609.421 ns |  4,669.222 ns |   0.70 |     0.05 |
| ParallelizedSimd |  100000 |    62,653.87 ns |  1,247.187 ns |  3,393.063 ns |   0.63 |     0.03 |
|       **Sequential** | **1000000** | **1,041,421.30 ns** |  **6,283.693 ns** |  **5,570.330 ns** |   **1.00** |     **0.00** |
|             Simd | 1000000 |   810,299.97 ns | 11,487.766 ns |  9,592.804 ns |   0.78 |     0.01 |
|     Parallelized | 1000000 |   419,496.96 ns |  8,763.994 ns | 25,004.168 ns |   0.40 |     0.02 |
| ParallelizedSimd | 1000000 |   400,718.96 ns |  7,924.591 ns | 18,208.066 ns |   0.38 |     0.02 |
