using Deedle;
using NumSharp;
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

        public void train(int n_a, int n_x, int n_y, int m, int num_iterations, double learning_rate, NDArray X, NDArray A0, bool print_cost)
        {
            double cost = 0;
            for (int i = 0; i < num_iterations; i++)
            {
                /*// Forward propagation
                NDArray a0 = X;
                var forward_values = rnn.lstm_forward(a0, A0);
                var a = forward_values[0];
                var y_hat = forward_values[1];
                var caches = forward_values[2];

                // Backward propagation
                var backward_values = rnn.lstm_backward(a, y_hat, caches);
                var gradients = backward_values[0];
                var da0 = backward_values[1];
                var dA0 = backward_values[2];

                // Update parameters
                var parameters = rnn.update_parameters(gradients, learning_rate);

                // Print the cost every 1000 iterations
                if (print_cost && i % 1000 == 0)
                {
                    Console.WriteLine("Cost after iteration " + i + ": " + cost);
                }
                if (print_cost && i % 100 == 0)
                {
                    costs.Add(cost);
                }*/
                
                
                
            }
        }
    }
}
