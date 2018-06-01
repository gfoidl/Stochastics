``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|             Method |       N |           Mean |         Error |        StdDev |
|------------------- |-------- |---------------:|--------------:|--------------:|
| **AverageAndVariance** |    **1000** |       **853.1 ns** |     **16.977 ns** |     **18.165 ns** |
|              Delta |    1000 |     1,755.6 ns |     32.540 ns |     30.438 ns |
|             MinMax |    1000 |       823.3 ns |      4.747 ns |      4.441 ns |
|   SkewnessKurtosis |    1000 |     2,595.4 ns |     15.532 ns |     14.528 ns |
|    ZTransformation |    1000 |     1,904.6 ns |     35.756 ns |     78.486 ns |
| **AverageAndVariance** | **1000000** |   **450,861.7 ns** |  **7,216.929 ns** |  **6,026.462 ns** |
|              Delta | 1000000 | 1,578,396.3 ns | 30,495.851 ns | 42,750.924 ns |
|             MinMax | 1000000 |   419,959.6 ns |  6,185.965 ns |  5,786.355 ns |
|   SkewnessKurtosis | 1000000 | 2,115,165.9 ns | 20,232.177 ns | 17,935.296 ns |
|    ZTransformation | 1000000 | 4,617,691.7 ns | 89,923.772 ns | 84,114.744 ns |
