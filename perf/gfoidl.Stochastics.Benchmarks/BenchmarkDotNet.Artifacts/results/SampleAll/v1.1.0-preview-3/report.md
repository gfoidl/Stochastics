``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.300-preview3-008416
  [Host]     : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT
  DefaultJob : .NET Core 2.1.0-preview2-26314-02 (Framework 4.6.26310.01), 64bit RyuJIT


```
|             Method |       N |           Mean |          Error |         StdDev |         Median |
|------------------- |-------- |---------------:|---------------:|---------------:|---------------:|
| **AverageAndVariance** |    **1000** |       **408.3 ns** |      **18.380 ns** |      **54.195 ns** |       **376.3 ns** |
|              Delta |    1000 |       712.4 ns |      18.238 ns |      24.347 ns |       710.8 ns |
|             MinMax |    1000 |       344.8 ns |      16.067 ns |      47.375 ns |       341.0 ns |
|   SkewnessKurtosis |    1000 |     1,005.0 ns |       2.101 ns |       1.966 ns |     1,004.8 ns |
|    ZTransformation |    1000 |     1,327.1 ns |      28.416 ns |      68.082 ns |     1,298.3 ns |
| **AverageAndVariance** | **1000000** |   **359,407.9 ns** |   **3,118.852 ns** |   **2,917.376 ns** |   **359,535.4 ns** |
|              Delta | 1000000 |   779,829.1 ns |  16,401.826 ns |  34,953.628 ns |   770,405.4 ns |
|             MinMax | 1000000 |   345,878.7 ns |  11,653.444 ns |  33,993.643 ns |   342,221.5 ns |
|   SkewnessKurtosis | 1000000 | 1,078,602.6 ns |  56,340.399 ns | 166,120.937 ns |   993,057.3 ns |
|    ZTransformation | 1000000 | 6,867,211.2 ns | 144,743.702 ns | 426,780.071 ns | 6,945,596.8 ns |
