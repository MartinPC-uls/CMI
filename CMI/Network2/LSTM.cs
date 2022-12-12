using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CMI.Utils;
using static CMI.Matrix;
using static CMI.Data.Normalizer;

namespace CMI.Network2
{
    public class LSTM
    {
        protected static double[] Wa { get; set; }
        protected static double[] Wi { get; set; }
        protected static double[] Wf { get; set; }
        protected static double[] Wo { get; set; }
        protected static double Ua { get; set; }
        protected static double Ui { get; set; }
        protected static double Uf { get; set; }
        protected static double Uo { get; set; }
        protected static double ba { get; set; }
        protected static double bi { get; set; }
        protected static double bf { get; set; }
        protected static double bo { get; set; }

        private List<Cell> Cells;

        protected static int Size;

        private static readonly double LEARNING_RATE = 0.3;

        private double[] xt { get; set; }

        public LSTM(int size)
        {
            Size = size;
        }
        public LSTM()
        {
        }

        public void InitializeWeights()
        {
            Wa = GenerateRandomMatrix(Size);
            Wi = GenerateRandomMatrix(Size);
            Wf = GenerateRandomMatrix(Size);
            Wo = GenerateRandomMatrix(Size);
            Ua = GenerateRandom();
            Ui = GenerateRandom();
            Uf = GenerateRandom();
            Uo = GenerateRandom();

            ba = GenerateRandom();
            bi = GenerateRandom();
            bf = GenerateRandom();
            bo = GenerateRandom();
        }
        
        public void Train(double[] inputs, double[] outputs, int number_of_iterations, int epochs = 1, double prev_long = 0, double prev_short = 0)
        {
            double aux_prev_long = prev_long;
            double aux_prev_short = prev_short;
            for (int i = 1; i <= number_of_iterations; i++)
            {
                Cells = new();
                for (int j = 0; j < inputs.Length; j++)
                {
                    Cell cell = new(inputs[j], prev_long, prev_short);
                    cell.Forward();
                    if (i % epochs == 0)
                    {
                        //Print(Denormalize(cell.ht));
                        Print(cell.ht);
                    }
                    prev_long = cell.ct;
                    prev_short = cell.ht;
                    Cells.Add(cell);
                }
                if (i % epochs == 0)
                {
                    Print("\n");
                    var total_loss = 0.0;
                    for (int z = 0; z < Cells.Count; z++)
                    {
                        total_loss += Math.Pow(Cells[z].ht - outputs[z], 2);
                    }
                    total_loss /= outputs.Length;
                    Print("Total loss: " + total_loss + "\n");
                }
                prev_long = aux_prev_long;
                prev_short = aux_prev_short;

                BackPropagationThroughTime(outputs, LEARNING_RATE);
                //if (i % epochs == 0)
                    //Print("\nTRAINING PROCESS: " + Math.Round(((((double)i + 1) / (double)number_of_iterations) * 100.0), 0) + "%", false);
            }
            // save parameters
        }

        private void BackPropagationThroughTime(double[] originalOutput, double learningRate)
        {
            double next_dht = 0;
            double next_dct = 0;
            double next_f = 0;
            for (int i = Cells.Count - 1; i >= 0; i--)
            {
                Cells[i].Backpropagation(originalOutput[i], next_dht, next_dct, next_f);
                next_dht = Cells[i].dht_1;
                next_dct = Cells[i].dct;
                next_f = Cells[i].f;
            }
            UpdateParameters(LEARNING_RATE);
        }

        private void UpdateParameters(double learningRate)
        {
            double dWa = 0;
            double dWi = 0;
            double dWf = 0;
            double dWo = 0;
            double dUa = 0;
            double dUi = 0;
            double dUf = 0;
            double dUo = 0;

            double dba = 0;
            double dbi = 0;
            double dbf = 0;
            double dbo = 0;
            
            // W and b
            for (int i = 0; i < Cells.Count; i++)
            {
                dWa += Cells[i].da * Cells[i].x;
                dWi += Cells[i].di * Cells[i].x;
                dWf += Cells[i].df * Cells[i].x;
                dWo += Cells[i].do_ * Cells[i].x;
                
                dba += Cells[i].da;
                dbi += Cells[i].di;
                dbf += Cells[i].df;
                dbo += Cells[i].do_;
            }
            // U
            for (int i = 0; i < Cells.Count - 1; i++)
            {
                dUa += Cells[i + 1].da * Cells[i].ht;
                dUi += Cells[i + 1].di * Cells[i].ht;
                dUf += Cells[i + 1].df * Cells[i].ht;
                dUo += Cells[i + 1].do_ * Cells[i].ht;
            }

            // Update weights of input
            //Wa -= learningRate * dWa;
            //Wi -= learningRate * dWi;
            //Wf -= learningRate * dWf;
            //Wo -= learningRate * dWo;

            // Update weights of hidden state
            Ua -= learningRate * dUa;
            Ui -= learningRate * dUi;
            Uf -= learningRate * dUf;
            Uo -= learningRate * dUo;

            // Update biases
            ba -= learningRate * dba;
            bi -= learningRate * dbi;
            bf -= learningRate * dbf;
            bo -= learningRate * dbo;
        }

        
    }   
}
