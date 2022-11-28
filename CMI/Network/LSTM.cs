namespace CMI.Network
{
    public class LSTM : Utils
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

        private static readonly double LEARNING_RATE = 0.5;

        private static readonly string PARAMETERS_FILE = AppDomain.CurrentDomain.BaseDirectory + "parameters.txt";

        private List<Cell> cells;

        public void loadParameters()
        {
            if (!File.Exists(PARAMETERS_FILE))
                throw new Exception("Parameters file not found.");

            string[] parameters = File.ReadAllLines(PARAMETERS_FILE);
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
        public void saveParameters()
        {
            if (File.Exists(PARAMETERS_FILE))
                File.Delete(PARAMETERS_FILE);

            File.Create(PARAMETERS_FILE);
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

        public void train(double[] inputs, double[] original_outputs, int number_of_iterations = 0, double prev_long = 0, double prev_short = 0)
        {
            double aux_prev_long = prev_long;
            double aux_prev_short = prev_short;
            for (int i = 1; i <= number_of_iterations; i++)
            {
                cells = new();
                for (int j = 0; j < inputs.Length; j++)
                {
                    Cell cell = new(inputs[j], prev_long, prev_short);
                    cell.forward();
                    if (i % 10000000 == 0)
                    {
                        print(cell.ht);
                    }
                    prev_long = cell.ct;
                    prev_short = cell.ht;
                    cells.Add(cell);
                }
                prev_long = aux_prev_long;
                prev_short = aux_prev_short;

                BackPropagationThroughTime(original_outputs, LEARNING_RATE);
            }

        }

        public void prediction(double[] inputs, double prev_long = 0, double prev_short = 0)
        {
            cells = new();
            for (int j = 0; j < inputs.Length; j++)
            {
                Cell cell = new(inputs[j], prev_long, prev_short);
                cell.forward();
                print(cell.ht);
                prev_long = cell.ct;
                prev_short = cell.ht;
                cells.Add(cell);
            }
            print("\n");
        }

        public LSTM()
        {
        }

        public void initialize()
        {
            Wa = generateRandom();
            Ua = generateRandom();
            ba = generateRandom();

            Wi = generateRandom();
            Ui = generateRandom();
            bi = generateRandom();

            Wf = generateRandom();
            Uf = generateRandom();
            bf = generateRandom();

            Wo = generateRandom();
            Uo = generateRandom();
            bo = generateRandom();
        }

        public void BackPropagationThroughTime(double[] original_output, double learning_rate)
        {
            double dht = 0;
            double next_dct = 0;
            double next_f = 0;
            for (int i = cells.Count - 1; i >= 0; i--)
            {
                cells[i].backpropagation(original_output[i], dht, next_dct, next_f);
                dht = cells[i].dht_1;
                next_dct = cells[i].dct;
                next_f = cells[i].f;
            }
            UpdateParameters(learning_rate);
        }

        public void UpdateParameters(double learning_rate)
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
    }
}
