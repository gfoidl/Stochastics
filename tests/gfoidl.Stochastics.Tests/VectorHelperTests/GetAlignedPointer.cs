using System.Numerics;
using System.Runtime.CompilerServices;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.VectorHelperTests
{
    [TestFixture]
    public unsafe class GetAlignedPointer
    {
        [Test]
        public void Natural_aligned_Pointer_given___Returns_SIMD_aligned_pointer()
        {
            var arr = new double[100];

            fixed (double* ptr = arr)
            {
                double* a   = ptr;
                double* end = VectorHelper.GetAlignedPointer(a);

                Assert.Multiple(() =>
                {
                    Assert.AreEqual(0, (long)end % Unsafe.SizeOf<Vector<double>>());
                    Assert.GreaterOrEqual(end - a, 0);
                });
            }
        }
        //---------------------------------------------------------------------
        [Test]
        public void Pointer_plus_1_given___Returns_SIMD_aligned_pointer()
        {
            var arr = new double[100];

            fixed (double* ptr = arr)
            {
                double* a   = ptr + 1;
                double* end = VectorHelper.GetAlignedPointer(a);

                Assert.Multiple(() =>
                {
                    Assert.AreEqual(0, (long)end % Unsafe.SizeOf<Vector<double>>());
                    Assert.GreaterOrEqual(end - a, 0);
                });
            }
        }
    }
}
