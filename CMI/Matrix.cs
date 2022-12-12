using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CMI.Utils;

namespace CMI
{
    public class Matrix
    {
        public static double Multiply(double[] a, double[] b)
        {
            double result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }
            return result;
        }

        public static double Multiply(double[,] a, double[,] b)
        {
            double result = 0;
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    result += a[i, j] * b[i, j];
                }
            }
            return result;
        }

        public static double[] HadamardProduct(double[] a, double[] b)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }

        public static double[,] HadamardProduct(double[,] a, double[,] b)
        {
            double[,] result = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
        }

        public static double Multiply(double[] a, double b)
        {
            double result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b;
            }
            return result;
        }

        public static double Multiply(double a, double[] b)
        {
            double result = 0;
            for (int i = 0; i < b.Length; i++)
            {
                result += a * b[i];
            }
            return result;
        }

        public static double[] Transpose(double[,] matrix)
        {
            double[] result = new double[matrix.GetLength(0) * matrix.GetLength(1)];
            int index = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    result[index] = matrix[i, j];
                    index++;
                }
            }
            return result;
        }

        public static double[,] Transpose(double[] matrix)
        {
            double[,] result = new double[matrix.Length, 1];
            for (int i = 0; i < matrix.Length; i++)
            {
                result[i, 0] = matrix[i];
            }
            return result;
        }

        public static double[] Pow(double[] matrix, double power)
        {
            double[] result = new double[matrix.Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                result[i] = Math.Pow(matrix[i], power);
            }
            return result;
        }

        public static double[] Substract(double number, double[] matrix)
        {
            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] = number - matrix[i];
            }
            return matrix;
        }
        
        public static double[] Substract(double[] matrix, double number)
        {
            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] -= number;
            }
            return matrix;
        }

        public static double InnerProduct(double[] matrix, double number)
        {
            double result = 0;
            for (int i = 0; i < matrix.Length; i++)
            {
                result += matrix[i] * number;
            }
            return result;
        }

        public static double InnerProduct(double number, double[] matrix)
        {
            return InnerProduct(matrix, number);
        }
    }
}
