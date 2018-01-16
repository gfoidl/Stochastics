``` ini

BenchmarkDotNet=v0.10.11, OS=ubuntu 16.04
Processor=Intel Xeon CPU 2.60GHz, ProcessorCount=2
.NET Core SDK=2.1.3
  [Host]	 : .NET Core 2.0.4 (Framework 4.6.0.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.4 (Framework 4.6.0.0), 64bit RyuJIT


```
|			  Method |		N |				Mean |			Error |			StdDev | Scaled | ScaledSD |
|------------------- |------- |-----------------:|---------------:|---------------:|-------:|---------:|
|	**UnsafeSequential** |	  **100** |			**4.623 us** |		**0.0924 us** |		 **0.1949 us** |   **1.00** |	  **0.00** |
|		  UnsafeSimd |	  100 |			2.208 us |		0.0111 us |		 0.0098 us |   0.48 |	  0.02 |
| UnsafeParallelSimd |	  100 |			9.147 us |		0.1064 us |		 0.0943 us |   1.98 |	  0.08 |
|	**UnsafeSequential** |	 **1000** |		  **412.731 us** |		**4.4867 us** |		 **3.5029 us** |   **1.00** |	  **0.00** |
|		  UnsafeSimd |	 1000 |		  223.901 us |		0.8551 us |		 0.7580 us |   0.54 |	  0.00 |
| UnsafeParallelSimd |	 1000 |		  192.020 us |		4.5770 us |		 4.8974 us |   0.47 |	  0.01 |
|	**UnsafeSequential** |	**10000** |	   **41,280.072 us** |	  **745.7709 us** |	   **582.2488 us** |   **1.00** |	  **0.00** |
|		  UnsafeSimd |	10000 |	   22,298.688 us |	  445.9051 us |	   950.2602 us |   0.54 |	  0.02 |
| UnsafeParallelSimd |	10000 |	   17,418.397 us |	   62.4781 us |		58.4420 us |   0.42 |	  0.01 |
|	**UnsafeSequential** | **100000** | **4,250,159.472 us** | **56,529.4527 us** | **47,204.6487 us** |   **1.00** |	  **0.00** |
|		  UnsafeSimd | 100000 | 2,410,592.785 us | 46,039.3799 us | 43,065.2604 us |   0.57 |	  0.01 |
| UnsafeParallelSimd | 100000 | 1,811,591.948 us |	2,317.5316 us |	 2,167.8203 us |   0.43 |	  0.00 |
