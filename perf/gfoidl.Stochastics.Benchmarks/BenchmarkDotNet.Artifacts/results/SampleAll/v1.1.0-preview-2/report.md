``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|             Method |       N |           Mean |         Error |        StdDev |
|------------------- |-------- |---------------:|--------------:|--------------:|
| **AverageAndVariance** |    **1000** |       **820.6 ns** |      **2.053 ns** |      **1.714 ns** |
|              Delta |    1000 |     1,589.7 ns |      6.956 ns |      5.808 ns |
|             MinMax |    1000 |       837.5 ns |     11.349 ns |     10.060 ns |
|   SkewnessKurtosis |    1000 |     1,908.2 ns |     53.261 ns |    157.040 ns |
|    ZTransformation |    1000 |     1,808.4 ns |     25.085 ns |     23.465 ns |
| **AverageAndVariance** | **1000000** |   **449,961.0 ns** |  **7,919.205 ns** |  **7,407.629 ns** |
|              Delta | 1000000 | 1,612,093.8 ns | 28,327.863 ns | 35,825.677 ns |
|             MinMax | 1000000 |   420,079.7 ns |  8,259.630 ns | 15,103.201 ns |
|   SkewnessKurtosis | 1000000 | 1,725,654.0 ns | 32,602.892 ns | 37,545.536 ns |
|    ZTransformation | 1000000 | 4,185,916.7 ns | 70,104.966 ns | 58,540.816 ns |
