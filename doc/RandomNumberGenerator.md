# RandomNumberGenerator

## Description

A simple (pseudo) random number generator.

* uniform distributed random numbers (Lehmer / Knuth)  
* normal distributed random numbers  
* exponential distributed random numbers

## Usage

```csharp
var rnd = new RandomNumberGenerator();

for (int i = 0; i < 100; ++i)
    Console.WriteLine($"{rnd.Uniform()}\t{rnd.NormalDistributed(0, 1)}\t{rnd.ExponentialDistributed(2.5)}");
```
