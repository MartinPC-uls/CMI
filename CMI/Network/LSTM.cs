using static CMI.Utils;
using static CMI.Data.Normalizer;

namespace CMI.Network
{
    public class LSTM
    {
        // W: Weight for input
        // U: Weight for hidden state
        // b: bias
        protected static double Wa, Ua;
        protected static double Wi, Ui;
        protected static double Wf, Uf;
        protected static double Wo, Uo;
        protected static double ba;
        protected static double bi;
        protected static double bf;
        protected static double bo;

        private static double prev_vdWa, prev_vdWi, prev_vdWf, prev_vdWo;
        private static double prev_vdUa, prev_vdUi, prev_vdUf, prev_vdUo;
        private static double prev_vdba, prev_vdbi, prev_vdbf, prev_vdbo;

        private static double prev_sdWa, prev_sdWi, prev_sdWf, prev_sdWo;
        private static double prev_sdUa, prev_sdUi, prev_sdUf, prev_sdUo;
        private static double prev_sdba, prev_sdbi, prev_sdbf, prev_sdbo;

        public List<char> Predicted_Output;

        private static readonly double LEARNING_RATE = 0.01;

        private static readonly string PARAMETERS_FILE = AppDomain.CurrentDomain.BaseDirectory + "parameters.txt";

        private List<Cell> cells;

        public void LoadParameters(string parameters_file = "")
        {
            if (parameters_file == "")
                parameters_file = PARAMETERS_FILE;

            if (!File.Exists(parameters_file))
                throw new Exception("Parameters file not found.");

            string[] parameters = File.ReadAllLines(parameters_file);
            if (parameters.Length < 12)
                throw new Exception("Could not read the file. Missing parameters.");

            Wa = double.Parse(parameters[0]);
            Wi = double.Parse(parameters[1]);
            Wf = double.Parse(parameters[2]);
            Wo = double.Parse(parameters[3]);

            Ua = double.Parse(parameters[4]);
            Ui = double.Parse(parameters[5]);
            Uf = double.Parse(parameters[6]);
            Uo = double.Parse(parameters[7]);

            ba = double.Parse(parameters[8]);
            bi = double.Parse(parameters[9]);
            bf = double.Parse(parameters[10]);
            bo = double.Parse(parameters[11]);
        }
        public void SaveParameters()
        {
            if (File.Exists(PARAMETERS_FILE))
                File.Delete(PARAMETERS_FILE);

            File.Create(PARAMETERS_FILE).Close();
            using var sw = new StreamWriter(PARAMETERS_FILE);
            sw.WriteLine(Wa);
            sw.WriteLine(Wi);
            sw.WriteLine(Wf);
            sw.WriteLine(Wo);

            sw.WriteLine(Ua);
            sw.WriteLine(Ui);
            sw.WriteLine(Uf);
            sw.WriteLine(Uo);

            sw.WriteLine(ba);
            sw.WriteLine(bi);
            sw.WriteLine(bf);
            sw.WriteLine(bo);
        }

        public void SeqPrediction(double[] inputs, int output_size, double prev_long = 0, double prev_short = 0)
        {
            Predicted_Output = new();
            double[] outputs = inputs;
            double[] aux = new double[outputs.Length];
            for (int i = 0; i < output_size; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    //print("in: " + Denormalize(outputs[j]));
                    Cell cell = new(outputs[j], prev_long, prev_short);
                    cell.Forward();
                    prev_long = cell.ct;
                    prev_short = cell.ht;
                    aux[j] = cell.ht;
                    //print("o: " + Denormalize(cell.ht));
                }
                Print(Denormalize(prev_short));
                Predicted_Output.Add(Denormalize(prev_short));
                prev_long = 0;
                prev_short = 0;

                outputs = aux;
            }
        }

        public void Prediction(double[] inputs, double prev_long = 0, double prev_short = 0)
        {
            Predicted_Output = new();
            for (int i = 0; i < inputs.Length; i++)
            {
                Cell cell = new(inputs[i], prev_long, prev_short);
                cell.Forward();
                prev_long = cell.ct;
                prev_short = cell.ht;
                Print(Denormalize(cell.ht));
                Predicted_Output.Add(Denormalize(cell.ht));
            }
        }

        public void SeqTrain(List<char[]> splitted_input, List<char[]> splitted_output, int number_of_iterations, int epochs = 1, double prev_long = 0, double prev_short = 0)
        {
            List<double[]> normalized_input = new();
            List<double[]> normalized_output = new();
            foreach (var input in splitted_input)
            {
                normalized_input.Add(Normalize(input));
                Print(Normalize(input));
            }
            foreach(var output in splitted_output)
            {
                normalized_output.Add(Normalize(output));
            }

            for (int i = 0; i < normalized_input.Count; i++)
            {
                Train(normalized_input[i], normalized_output[i], number_of_iterations, epochs, prev_long, prev_short);
                //if (i)
                //Print("\nTRAINING PROCESS: " + Math.Round(((((double)i + 1) / (double)normalized_input.Count) * 100.0), 0) + "%", false);
            }

            Print("\nTraining process completed.\n");
        }

        public void Train(double[] inputs, double[] outputs, int number_of_iterations, int epochs = 1, double prev_long = 0, double prev_short = 0)
        {
            double aux_prev_long = prev_long;
            double aux_prev_short = prev_short;
            for (int i = 1; i <= number_of_iterations; i++)
            {
                cells = new();
                if (i % epochs == 0)
                    Print("\n");
                for (int j = 0; j < inputs.Length; j++)
                {
                    Cell cell = new(inputs[j], prev_long, prev_short);
                    cell.Forward();
                    if (i % epochs == 0)
                    {
                        Print(Denormalize(cell.ht), false);
                        //Print(cell.ht);
                    }
                    prev_long = cell.ct;
                    prev_short = cell.ht;
                    cells.Add(cell);
                }
                prev_long = aux_prev_long;
                prev_short = aux_prev_short;

                BackPropagationThroughTime(outputs, LEARNING_RATE);
                if (i % epochs == 0)
                {
                    Print("\n");
                    var total_loss = 0.0;
                    for (int z = 0; z < cells.Count; z++)
                    {
                        total_loss += Math.Pow(cells[z].ht - outputs[z], 2);
                    }
                    total_loss /= (double)outputs.Length;
                    Print("TRAINING PROCESS: " + Math.Round(((((double)i + 1) / (double)number_of_iterations) * 100.0), 0) + "%\tloss: " + 
                        Decimal.Parse(total_loss.ToString(), System.Globalization.NumberStyles.AllowExponent | System.Globalization.NumberStyles.AllowDecimalPoint),
                        false);
                }
            }
            SaveParameters();
        }

        public LSTM()
        {
            prev_vdWa = 0;
            prev_vdWi = 0;
            prev_vdWf = 0;
            prev_vdWo = 0;

            prev_vdUa = 0;
            prev_vdUi = 0;
            prev_vdUf = 0;
            prev_vdUo = 0;
            
            prev_vdba = 0;
            prev_vdbi = 0;
            prev_vdbf = 0;
            prev_vdbo = 0;

            prev_sdWa = 0;
            prev_sdWi = 0;
            prev_sdWf = 0;
            prev_sdWo = 0;

            prev_sdUa = 0;
            prev_sdUi = 0;
            prev_sdUf = 0;
            prev_sdUo = 0;

            prev_sdba = 0;
            prev_sdbi = 0;
            prev_sdbf = 0;
            prev_sdbo = 0;
        }

        public void initialize()
        {
            Wa = GenerateRandom();
            Ua = GenerateRandom();
            ba = 0;
                         
            Wi = GenerateRandom();
            Ui = GenerateRandom();
            bi = 0;
                         
            Wf = GenerateRandom();
            Uf = GenerateRandom();
            bf = 0;
                         
            Wo = GenerateRandom();
            Uo = GenerateRandom();
            bo = 0;
        }

        private void BackPropagationThroughTime(double[] original_output, double learning_rate)
        {
            double next_dht = 0;
            double next_dct = 0;
            double next_f = 0;
            for (int i = cells.Count - 1; i >= 0; i--)
            {
                cells[i].Backpropagation(original_output[i], next_dht, next_dct, next_f);
                next_dht = cells[i].dht_1;
                next_dct = cells[i].dct;
                next_f = cells[i].f;
            }
            //UpdateParameters(learning_rate);
            AdamOptimizer(learning_rate);
        }
        
        private void UpdateParameters(double learning_rate)
        {
            double dWa = 0, dUa = 0, dba = 0;
            double dWi = 0, dUi = 0, dbi = 0;
            double dWf = 0, dUf = 0, dbf = 0;
            double dWo = 0, dUo = 0, dbo = 0;

            // W and b
            for (int i = 0; i < cells.Count; i++)
            {
                dWa += cells[i].da * cells[i].x;
                dWi += cells[i].di * cells[i].x;
                dWf += cells[i].df * cells[i].x;
                dWo += cells[i].do_ * cells[i].x;

                dba += cells[i].da;
                dbi += cells[i].di;
                dbf += cells[i].df;
                dbo += cells[i].do_;
            }
            // U
            for (int i = 0; i < cells.Count - 1; i++)
            {
                dUa += cells[i + 1].da * cells[i].ht;
                dUi += cells[i + 1].di * cells[i].ht;
                dUf += cells[i + 1].df * cells[i].ht;
                dUo += cells[i + 1].do_ * cells[i].ht;
            }

            // Update weights of input
            Wa -= learning_rate * dWa;
            Wi -= learning_rate * dWi;
            Wf -= learning_rate * dWf;
            Wo -= learning_rate * dWo;

            // Update weights of hidden state
            Ua -= learning_rate * dUa;
            Ui -= learning_rate * dUi;
            Uf -= learning_rate * dUf;
            Uo -= learning_rate * dUo;

            // Update biases
            ba -= learning_rate * dba;
            bi -= learning_rate * dbi;
            bf -= learning_rate * dbf;
            bo -= learning_rate * dbo;
        }

        public void AdamOptimizer(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 0.00000001)
        {
            double dWa = 0, dUa = 0, dba = 0;
            double dWi = 0, dUi = 0, dbi = 0;
            double dWf = 0, dUf = 0, dbf = 0;
            double dWo = 0, dUo = 0, dbo = 0;

            // W and b
            for (int i = 0; i < cells.Count; i++)
            {
                dWa += cells[i].da * cells[i].x;
                dWi += cells[i].di * cells[i].x;
                dWf += cells[i].df * cells[i].x;
                dWo += cells[i].do_ * cells[i].x;

                dba += cells[i].da;
                dbi += cells[i].di;
                dbf += cells[i].df;
                dbo += cells[i].do_;
            }
            // U
            for (int i = 0; i < cells.Count - 1; i++)
            {
                dUa += cells[i + 1].da * cells[i].ht;
                dUi += cells[i + 1].di * cells[i].ht;
                dUf += cells[i + 1].df * cells[i].ht;
                dUo += cells[i + 1].do_ * cells[i].ht;
            }

            double min = -1;
            double max = 1;
            
            dWa = ClipGradient(dWa, min, max, true);
            dWi = ClipGradient(dWi, min, max, true);
            dWf = ClipGradient(dWf, min, max, true);
            dWo = ClipGradient(dWo, min, max, true);

            dUa = ClipGradient(dUa, min, max, true);
            dUi = ClipGradient(dUi, min, max, true);
            dUf = ClipGradient(dUf, min, max, true);
            dUo = ClipGradient(dUo, min, max, true);

            dba = ClipGradient(dba, min, max, true);
            dbi = ClipGradient(dbi, min, max, true);
            dbf = ClipGradient(dbf, min, max, true);
            dbo = ClipGradient(dbo, min, max, true);
            /*Print("dWa: " + dWa);
            Print("dWi: " + dWi);
            Print("dWf: " + dWf);
            Print("dWo: " + dWo);

            Print("dUa: " + dUa);
            Print("dUi: " + dUi);
            Print("dUf: " + dUf);
            Print("dUo: " + dUo);

            Print("dba: " + dba);
            Print("dbi: " + dbi);
            Print("dbf: " + dbf);
            Print("dbo: " + dbo);*/

            var vdWa = beta1 * prev_vdWa + (1 - beta1) * dWa;
            var vdWi = beta1 * prev_vdWi + (1 - beta1) * dWi;
            var vdWf = beta1 * prev_vdWf + (1 - beta1) * dWf;
            var vdWo = beta1 * prev_vdWo + (1 - beta1) * dWo;

            var vdUa = beta1 * prev_vdUa + (1 - beta1) * dUa;
            var vdUi = beta1 * prev_vdUi + (1 - beta1) * dUi;
            var vdUf = beta1 * prev_vdUf + (1 - beta1) * dUf;
            var vdUo = beta1 * prev_vdUo + (1 - beta1) * dUo;

            var vdba = beta1 * prev_vdba + (1 - beta1) * dba;
            var vdbi = beta1 * prev_vdbi + (1 - beta1) * dbi;
            var vdbf = beta1 * prev_vdbf + (1 - beta1) * dbf;
            var vdbo = beta1 * prev_vdbo + (1 - beta1) * dbo;

            var sdWa = beta2 * prev_sdWa + (1 - beta2) * Math.Pow(dWa, 2);
            var sdWi = beta2 * prev_sdWi + (1 - beta2) * Math.Pow(dWi, 2);
            var sdWf = beta2 * prev_sdWf + (1 - beta2) * Math.Pow(dWf, 2);
            var sdWo = beta2 * prev_sdWo + (1 - beta2) * Math.Pow(dWo, 2);

            var sdUa = beta2 * prev_sdUa + (1 - beta2) * Math.Pow(dUa, 2);
            var sdUi = beta2 * prev_sdUi + (1 - beta2) * Math.Pow(dUi, 2);
            var sdUf = beta2 * prev_sdUf + (1 - beta2) * Math.Pow(dUf, 2);
            var sdUo = beta2 * prev_sdUo + (1 - beta2) * Math.Pow(dUo, 2);

            var sdba = beta2 * prev_sdba + (1 - beta2) * Math.Pow(dba, 2);
            var sdbi = beta2 * prev_sdbi + (1 - beta2) * Math.Pow(dbi, 2);
            var sdbf = beta2 * prev_sdbf + (1 - beta2) * Math.Pow(dbf, 2);
            var sdbo = beta2 * prev_sdbo + (1 - beta2) * Math.Pow(dbo, 2);

            // Updating weights of input
            Wa -= learning_rate * (vdWa / Math.Sqrt(sdWa + epsilon));
            Wi -= learning_rate * (vdWi / Math.Sqrt(sdWi + epsilon));
            Wf -= learning_rate * (vdWf / Math.Sqrt(sdWf + epsilon));
            Wo -= learning_rate * (vdWo / Math.Sqrt(sdWo + epsilon));

            // Updating weights of hidden state
            Ua -= learning_rate * (vdUa / Math.Sqrt(sdUa + epsilon));
            Ui -= learning_rate * (vdUi / Math.Sqrt(sdUi + epsilon));
            Uf -= learning_rate * (vdUf / Math.Sqrt(sdUf + epsilon));
            Uo -= learning_rate * (vdUo / Math.Sqrt(sdUo + epsilon));

            // Updating biases
            ba -= learning_rate * (vdba / Math.Sqrt(sdba + epsilon));
            bi -= learning_rate * (vdbi / Math.Sqrt(sdbi + epsilon));
            bf -= learning_rate * (vdbf / Math.Sqrt(sdbf + epsilon));
            bo -= learning_rate * (vdbo / Math.Sqrt(sdbo + epsilon));

            prev_vdWa = vdWa;
            prev_vdWi = vdWi;
            prev_vdWf = vdWf;
            prev_vdWo = vdWo;

            prev_vdUa = vdUa;
            prev_vdUi = vdUi;
            prev_vdUf = vdUf;
            prev_vdUo = vdUo;

            prev_vdba = vdba;
            prev_vdbi = vdbi;
            prev_vdbf = vdbf;
            prev_vdbo = vdbo;

            prev_sdWa = sdWa;
            prev_sdWi = sdWi;
            prev_sdWf = sdWf;
            prev_sdWo = sdWo;

            prev_sdUa = sdUa;
            prev_sdUi = sdUi;
            prev_sdUf = sdUf;
            prev_sdUo = sdUo;

            prev_sdba = sdba;
            prev_sdbi = sdbi;
            prev_sdbf = sdbf;
            prev_sdbo = sdbo;
        }

        private double ClipGradient(double gradient, double min, double max, bool normalization)
        {
            if (gradient > max)
            {
                gradient = max;
            }
            else if (gradient < min)
            {
                gradient = min;
            }

            if (normalization)
            {
                gradient /= max;
            }

            return gradient;
        }
    }
}
