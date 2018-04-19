# Sample

## Description

Represents position, scatter and shape parameters of sample data.

The following parameters can be obtained:

* Sample size  
* [Mean / arithmetic average](https://en.wikipedia.org/wiki/Mean)  
* [Median](https://en.wikipedia.org/wiki/Median)  
* Maximum  
* Minimum  
* Range  
* [Delta -- mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)  
* [Standard deviation `1/N`](https://en.wikipedia.org/wiki/Standard_deviation)  
* [Sample standard deviation `1/(N-1)`](https://en.wikipedia.org/wiki/Standard_deviation)  
* [Variance `1/N`](https://en.wikipedia.org/wiki/Variance)  
* [Sample Variance `1/(N-1)`](https://en.wikipedia.org/wiki/Variance)  
* [Skewness](https://en.wikipedia.org/wiki/Skewness)  
* [Kurtosis](https://en.wikipedia.org/wiki/Kurtosis)  

As methods:

* [z-Transformation](https://en.wikipedia.org/wiki/Standard_score)  
* [Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation)  

## Usage

```csharp
double[] values = new double[100]
var rnd = new Random();

for (int i = 0; i < value.Length; ++i)
    values[i] = rnd.NextDouble();

var sample = new Sample(values);

double sigma = sample.Sigma;

Console.WriteLine(sample);
```

`ToString` is overwritten, therefore the output might be something like:
```
Count                  : 100
Mean                   : 0.519275519488973
Median                 : 0.51324658655247
Max                    : 0.983533758662424
Min                    : 0.015593185562451
Range                  : 0.967940573099973
Delta                  : 0.225629180047395
Sigma                  : 0.270445786115599
StandardDeviation      : 0.269090159603445
SampleStandardDeviation: 0.270445786115599
Variance               : 0.0724095139954074
SampleVariance         : 0.0731409232276842
Skewness               : -0.192630733586537
Kurtosis               : 2.05350514378756
```
