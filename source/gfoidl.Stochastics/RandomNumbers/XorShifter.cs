using System;
using System.Collections.Generic;
using System.Text;

namespace gfoidl.Stochastics.RandomNumbers
{
    public  class XorShifter
    {
        private uint _x32 = 314159265;
        private ulong _x64 = 88172645463325252;
        //---------------------------------------------------------------------
        public uint XorShift32()
        {
            uint x32 = _x32;

            x32 ^= x32 << 13;
            x32 ^= x32 >> 17;
            x32 ^= x32 << 5;

            return _x32 = x32;
        }
        //---------------------------------------------------------------------
        public ulong XorShift64()
        {
            ulong x64 = _x64;

            x64 ^= x64 << 13;
            x64 ^= x64 >> 7;
            x64 ^= x64 << 17;

            return _x64 = x64;
        }
    }
}
