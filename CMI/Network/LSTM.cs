using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace CMI.Network
{
    public class LSTM : Utils
    {
        protected static double Wsf { get; set; } // Weight of short-term memory to forget gate
        protected static double Wif { get; set; } // Weight of input to forget gate
        protected static double bf { get; set; } // Bias of forget
        
        protected static double Wipltm { get; set; } // Weight of input to potential long-term memory
        protected static double Wspltm { get; set; } // Weight of short-term memory to potential long-term memory
        protected static double bpltm { get; set; } // Bias of potential long-term 
        
        protected static double Wipmr { get; set; } // Weight of input to potential memory to remember
        protected static double Wspmr { get; set; } // Weight of short-term memory to potential memory to remember
        protected static double bpmr { get; set; } // Bias of potential memory to remember
        
        protected static double Wso { get; set; } // Weight of short-term memory to output gate
        protected static double Wio { get; set; } // Weight of input to output gate
        protected static double bo { get; set; } // Bias of output gate

        protected static double test { get; set; }

        private List<Cell> cells;

        public void train(double[] inputs, double[] original_output, int number_of_iterations = 0, double prev_long = 0, double prev_short = 0)
        {

            for (int i = 1; i <= number_of_iterations; i++)
            {
                cells = new List<Cell>();

                for (int j = 0; j < inputs.Length; j++)
                {
                    Cell newCell = new(inputs[j], prev_long, prev_short);
                    prev_long = newCell.next_long;
                    prev_short = newCell.next_short;
                    print(newCell.output);
                    cells.Add(newCell);
                }
                print("\n");


                BackPropagationThroughTime(original_output, 0.0005);
            }

        }

        public double mean_squared_error(double[] original_output, double[] predicted_output)
        {
            double sum = 0;
            for (int i = 0; i < original_output.Length; i++)
            {
                sum += Math.Pow(original_output[i] - predicted_output[i], 2);
            }
            print("loss: " + Math.Round((sum / original_output.Length), 3));
            return Math.Round((sum / original_output.Length), 3);
        }

        public LSTM()
        {
        }

        public void initialize()
        {
            test = np.random.randn(1).ToArray<double>()[0];

            Wsf = np.random.randn(1).ToArray<double>()[0];
            Wif = np.random.randn(1).ToArray<double>()[0];
            bf = np.random.randn(1).ToArray<double>()[0];

            Wipltm = np.random.randn(1).ToArray<double>()[0];
            Wspltm = np.random.randn(1).ToArray<double>()[0];
            bpltm = np.random.randn(1).ToArray<double>()[0];
            Wipmr = np.random.randn(1).ToArray<double>()[0];
            Wspmr = np.random.randn(1).ToArray<double>()[0];
            bpmr = np.random.randn(1).ToArray<double>()[0];

            Wso = np.random.randn(1).ToArray<double>()[0];
            Wio = np.random.randn(1).ToArray<double>()[0];
            bo = np.random.randn(1).ToArray<double>()[0];
        }

        public void BackPropagationThroughTime(double[] original_output, double learning_rate)
        {
            for (int i = cells.Count - 1; i >= 0; i--)
            {
                List<double> gradients = cells[i].backpropagation(original_output[2], cells[2].output);
                Wif -= learning_rate * gradients[0];
                //print("Wif: " + Wif);
                //print("\n");
                //print("[" + original_output[i] + ", " + cells[i].output + "]");
                //print("Wif: " + Wif);
                //print("Result: " + learning_rate * gradients[0]);
                //print("\n");
                Wsf -= learning_rate * gradients[1];
                bf -= learning_rate * gradients[2];
                Wipmr -= learning_rate * gradients[3];
                Wspmr -= learning_rate * gradients[4];
                bpmr -= learning_rate * gradients[5];
                Wipltm -= learning_rate * gradients[6];
                Wspltm -= learning_rate * gradients[7];
                bpltm -= learning_rate * gradients[8];
                Wio -= learning_rate * gradients[9];
                Wso -= learning_rate * gradients[10];
                bo -= learning_rate * gradients[11];
            }
        }
    }
}
