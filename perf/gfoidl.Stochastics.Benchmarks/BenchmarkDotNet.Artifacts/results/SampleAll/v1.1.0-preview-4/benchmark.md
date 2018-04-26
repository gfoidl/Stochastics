``` ini

BenchmarkDotNet=v0.10.11, OS=debian 9
Processor=Intel Xeon CPU E5-2680 v2 2.80GHz, ProcessorCount=32
.NET Core SDK=2.1.300-preview2-008530
  [Host]     : .NET Core 2.1.0-preview2-26406-04 (Framework 4.6.26406.07), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26406-04 (Framework 4.6.26406.07), 64bit RyuJIT


```
|             Method |       N |           Mean |          Error |         StdDev |
|------------------- |-------- |---------------:|---------------:|---------------:|
| **AverageAndVariance** |    **1000** |       **409.0 ns** |      **5.2029 ns** |      **4.8668 ns** |
|              Delta |    1000 |       791.7 ns |     13.9946 ns |     13.0905 ns |
|             MinMax |    1000 |       387.4 ns |      0.9558 ns |      0.8941 ns |
|   SkewnessKurtosis |    1000 |       966.8 ns |     10.1665 ns |      9.5097 ns |
|    ZTransformation |    1000 |     1,356.4 ns |      3.8765 ns |      3.2370 ns |
| **AverageAndVariance** | **1000000** |   **392,833.0 ns** |  **2,592.9602 ns** |  **2,425.4563 ns** |
|              Delta | 1000000 |   804,517.3 ns |    178.0478 ns |    157.8348 ns |
|             MinMax | 1000000 |   365,674.3 ns |  2,362.1774 ns |  2,209.5820 ns |
|   SkewnessKurtosis | 1000000 |   914,639.6 ns |    565.1503 ns |    471.9260 ns |
|    ZTransformation | 1000000 | 4,561,316.9 ns | 77,855.2961 ns | 72,825.8852 ns |
