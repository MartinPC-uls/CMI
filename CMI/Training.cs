using Deedle;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CMI
{
    public class Training
    {
        public RNN rnn;
        public Training(RNN rnn, int n_a)
        {
            this.rnn = rnn;
            var path = @"C:\Users\User\Desktop\asd.csv";
            var data = Frame.ReadCsv(path);
        }

        public void train(int n_a, int n_x, int n_y, int m, int num_iterations, double learning_rate, double[] X, double[] Y, bool print_cost)
        {
            double cost = 0;
            for (int i = 0; i < num_iterations; i++)
            {
                // Forward propagation
                double[] a0 = X;
                //double[] a1 = rnn.lstm_forward(a0, );
            }
        }
    }
}
