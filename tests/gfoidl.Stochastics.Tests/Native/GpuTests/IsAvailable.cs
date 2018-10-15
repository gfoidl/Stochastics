using System;
using gfoidl.Stochastics.Native;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Native.GpuTests
{
    [TestFixture, Explicit("Mutual exclusive tests")]
    public class IsAvailable
    {
        [Test]
        public void Env_set_to_0___false()
        {
            Environment.SetEnvironmentVariable(Gpu.EnvVariableName, "0");

            Assert.IsFalse(Gpu.IsAvailable);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Env_set_to_force___true()
        {
            Environment.SetEnvironmentVariable(Gpu.EnvVariableName, "force");

            Assert.IsTrue(Gpu.IsAvailable);
        }
    }
}
