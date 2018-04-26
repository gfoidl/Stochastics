``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|             Method |       N |           Mean |         Error |        StdDev |
|------------------- |-------- |---------------:|--------------:|--------------:|
| **AverageAndVariance** |    **1000** |       **954.2 ns** |     **23.826 ns** |     **22.287 ns** |
|              Delta |    1000 |     1,624.5 ns |      5.154 ns |      4.821 ns |
|             MinMax |    1000 |       825.7 ns |      3.333 ns |      3.117 ns |
|   SkewnessKurtosis |    1000 |     2,601.5 ns |      7.355 ns |      6.880 ns |
|    ZTransformation |    1000 |     1,917.7 ns |     32.959 ns |     42.855 ns |
| **AverageAndVariance** | **1000000** |   **450,323.6 ns** |  **6,148.518 ns** |  **5,450.501 ns** |
|              Delta | 1000000 | 1,587,060.1 ns | 30,802.382 ns | 48,855.842 ns |
|             MinMax | 1000000 |   419,121.1 ns |  7,115.256 ns |  6,307.489 ns |
|   SkewnessKurtosis | 1000000 | 2,148,394.0 ns | 40,587.797 ns | 39,862.640 ns |
|    ZTransformation | 1000000 | 4,634,365.1 ns | 87,140.186 ns | 81,510.976 ns |
