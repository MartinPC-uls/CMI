using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace CMI
{
    public class Matrix
    {
        public static double Dot(double[,] a, double[,] b)
        {
            double[,] c = new double[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    for (int k = 0; k < a.GetLength(1); k++)
                    {
                        c[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return c[0, 0];
        }

        public static NDArray GenerateRandomWeightMatrix(int x, int y)
        {
            return np.random.randn(x, y);
        }

        public static NDArray GenerateRandomBiasMatrix(int y)
        {
            return np.random.randn(1, y);
        }

        public static void Transpose(ref NDArray a)
        {
            a = a.T;
        }
    }
}
