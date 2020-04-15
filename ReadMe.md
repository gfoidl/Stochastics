| CircleCi | AppVeyor | Code Coverage | NuGet |
| -- | -- | -- | -- |
| [![CircleCI](https://circleci.com/gh/gfoidl/Stochastics/tree/master.svg?style=svg)](https://circleci.com/gh/gfoidl/Stochastics/tree/master) | [![Build status](https://ci.appveyor.com/api/projects/status/a0r3j3rygrwg4nx4/branch/master?svg=true)](https://ci.appveyor.com/project/GntherFoidl/stochastics/branch/master) | [![codecov](https://codecov.io/gh/gfoidl/Stochastics/branch/master/graph/badge.svg)](https://codecov.io/gh/gfoidl/Stochastics) | [![NuGet](https://img.shields.io/nuget/v/gfoidl.Stochastics.svg?style=flat-square)](https://www.nuget.org/packages/gfoidl.Stochastics/) |

# gfoidl.Stochastics

Current release: [v1.1.0 release notes](doc/release-notes/v1.1.0.md)

## Random numbers

* [(Pseudo) random numbers](doc/RandomNumberGenerator.md)

## Statistics

* [Sample](doc/Statistics/Sample.md)  

### Outlier Detection

* Outlier detection based on [Chauvenet's criterion](doc/Statistics/OutlierDetection/ChauvenetOutlierDetection.md)


## Development channel

To get packages from the development channel use a `nuget.config` similar to this one:
```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
    <packageSources>
        <add key="gfoidl-public" value="https://pkgs.dev.azure.com/gh-gfoidl/github-Projects/_packaging/gfoidl-public/nuget/v3/index.json" />
    </packageSources>
</configuration>
```
