﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace CMI
{
    public class RNN
    {
        NDArray Wax;
        NDArray Waa;
        NDArray Wya;
        NDArray ba;
        NDArray by;

        // lstm model variables
        NDArray Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy; // + by
        List<NDArray> parameters = new List<NDArray>();
        List<NDArray> parameters_lstm = new List<NDArray>();

        public RNN(List<NDArray> parameters)
        {
            this.parameters = parameters;
            Wax = parameters[0];
            Waa = parameters[1];
            Wya = parameters[2];
            ba = parameters[3];
            by = parameters[4];
        }

        public RNN(List<NDArray> parameters_lstm, int anyValue)
        {
            this.parameters_lstm = parameters_lstm;
            Wf = parameters_lstm[0];
            bf = parameters_lstm[1];
            Wi = parameters_lstm[2];
            bi = parameters_lstm[3];
            Wc = parameters_lstm[4];
            bc = parameters_lstm[5];
            Wo = parameters_lstm[6];
            bo = parameters_lstm[7];
            Wy = parameters_lstm[8];
            by = parameters_lstm[9];
        }

        public Tuple<NDArray, NDArray, Cache> rnn_cell_forward(NDArray xt, NDArray a_prev)
        {
            NDArray a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba);
            NDArray yt_pred = softmax(np.dot(Wya, a_next) + by);

            //Cache cache = new Cache(yt_pred, a_next, a_prev, parameters);
            Cache cache = new Cache(a_next, a_prev, xt, parameters);

            Tuple<NDArray, NDArray, Cache> values =
                new Tuple<NDArray, NDArray, Cache>(a_next, yt_pred, cache);

            return values;
        }

        public Tuple<NDArray, NDArray, NDArray, Cache> lstm_cell_forward(NDArray xt, NDArray a_prev, NDArray c_prev)
        {
            int n_x, m;
            n_x = xt.Shape[0];
            m = xt.Shape[1];

            int n_y, n_a;
            n_y = Wy.Shape[0];
            n_a = Wy.Shape[1];

            var concat = np.zeros((n_a + n_x, m));
            //Console.WriteLine("concat.Shape[0]: " + concat.Shape[0]);
            //Console.WriteLine("concat.Shape[1]: " + concat.Shape[1]);

            //Console.WriteLine("a_prev.Shape[0]: " + a_prev.Shape[0]);
            //Console.WriteLine("a_prev.Shape[1]: " + a_prev.Shape[1]);

            _replace(ref concat, a_prev, n_a);
            replace(ref concat, xt, n_a, true);

            var ft = sigmoid(np.dot(Wf, concat) + bf);
            var it = sigmoid(np.dot(Wi, concat) + bi);
            var cct = np.tanh(np.dot(Wc, concat) + bc);
            var c_next = np.multiply(ft, c_prev) + np.multiply(it, cct);
            var ot = sigmoid(np.dot(Wo, concat) + bo);
            var a_next = np.multiply(ot, np.tanh(c_next));
            
            var yt_pred = softmax(np.dot(Wy, a_next) + by);

            //Cache cache = new(a_next, a_prev, yt_pred, c_next, c_prev, ft, it, cct, ot, parameters_lstm); // is it correct? review this
            Cache cache = new(a_next, a_prev, xt, c_next, c_prev, ft, it, cct, ot, parameters_lstm, yt_pred);

            Tuple<NDArray, NDArray, NDArray, Cache> values =
                new Tuple<NDArray, NDArray, NDArray, Cache>(a_next, c_next, yt_pred, cache);

            return values;
        }

        public Tuple<NDArray, NDArray, Tuple<List<Cache>, NDArray>> rnn_forward(NDArray x, NDArray a0)
        {
            List<Cache> caches = new List<Cache>();
            int n_x, m, T_x;
            n_x = x.Shape[0];
            m = x.Shape[1];
            T_x = x.Shape[2];

            int n_y, n_a;
            n_y = Wya.Shape[0];
            n_a = Wya.Shape[1];

            NDArray a = np.zeros((n_a, m, T_x));
            NDArray y_pred = np.zeros((n_y, m, T_x));

            NDArray a_next = a0;

            for (int t = 0; t < T_x; t++)
            {
                NDArray x_sliced = slice(x, t);
                //Console.WriteLine(x_sliced.ToString());

                Tuple<NDArray, NDArray, Cache> valuess = rnn_cell_forward(x_sliced, a_next);

                a_next = valuess.Item1;
                var yt_pred = valuess.Item2;

                //Console.WriteLine(a.ToString());
                //Console.WriteLine("replacing with:");
                replace(ref a, a_next, t);
                //Console.WriteLine(t);
                //Console.WriteLine(a.ToString());
                replace(ref y_pred, yt_pred, t);
                //Console.WriteLine("---");
                //Console.WriteLine(y_pred.ToString());
                caches.Add(valuess.Item3);
            }

            Tuple<NDArray, NDArray, Tuple<List<Cache>, NDArray>> values =
                new Tuple<NDArray, NDArray, Tuple<List<Cache>, NDArray>>(a, y_pred, new Tuple<List<Cache>, NDArray>(caches, x));
            return values;
        }

        public Tuple<NDArray, NDArray, NDArray, Tuple<List<Cache>, NDArray>> lstm_forward(NDArray x, NDArray a0)
        {
            List<Cache> caches = new List<Cache>();
            int n_x, m, T_x;
            n_x = x.Shape[0];
            m = x.Shape[1];
            T_x = x.Shape[2];

            int n_y, n_a;
            n_y = Wy.Shape[0];
            n_a = Wy.Shape[1];

            NDArray a = np.zeros((n_a, m, T_x));
            NDArray y_pred = np.zeros((n_y, m, T_x));
            NDArray c = np.zeros((n_a, m, T_x));

            NDArray a_next = a0;
            NDArray c_next = np.zeros((n_a, m));

            for (int t = 0; t < T_x; t++)
            {
                NDArray x_sliced = slice(x, t);
                //Console.WriteLine(x_sliced.ToString());

                Tuple<NDArray, NDArray, NDArray, Cache> valuess = lstm_cell_forward(x_sliced, a_next, c_next);

                a_next = valuess.Item1;
                c_next = valuess.Item2;
                var yt_pred = valuess.Item3;

                //Console.WriteLine(a.ToString());
                //Console.WriteLine("replacing with:");
                replace(ref a, a_next, t);
                //Console.WriteLine("a");
                //Console.WriteLine(slice(a, t).ToString());
                //Console.WriteLine(t);
                //Console.WriteLine(a.ToString());
                replace(ref y_pred, yt_pred, t);
                //Console.WriteLine("y_pred");
                //Console.WriteLine(y_pred.ToString());
                replace(ref c, c_next, t);
                //Console.WriteLine("---");
                //Console.WriteLine(y_pred.ToString());
                caches.Add(valuess.Item4);
            }

            Tuple<NDArray, NDArray, NDArray, Tuple<List<Cache>, NDArray>> values =
                new Tuple<NDArray, NDArray, NDArray, Tuple<List<Cache>, NDArray>>(a, y_pred, c, new Tuple<List<Cache>, NDArray>(caches, x));
            return values;
        }

        public List<NDArray> rnn_cell_backward(NDArray da_next, Cache cache)
        {
            //var (a_next, a_prev, xt, parameters) = cache;
            var a_next = cache.a_next;
            var a_prev = cache.a_prev;
            var xt = cache.xt;

            //var (Wax, Waa, Wya, ba, by) = parameters;
            var Wax = cache.Wax;
            var Waa = cache.Waa;
            var Wya = cache.Wya;
            var ba = cache.ba;
            var by = cache.by;

            var dtanh = (1 - np.power(a_next, 2)) * da_next;

            var dxt = np.dot(Wax.T, dtanh);
            var dWax = np.dot(dtanh, xt.T);

            var da_prev = np.dot(Waa.T, dtanh);
            var dWaa = np.dot(dtanh, a_prev.T);

            //var dba = np.sum(dtanh, axis: 1, keepdims: true);
            //var dba = np.sum(dtanh, axis: 1);
            var dba = sum(dtanh, 1, true);

            List<NDArray> gradients = new List<NDArray>();
            gradients.Add(dxt);
            gradients.Add(da_prev);
            gradients.Add(dWax);
            gradients.Add(dWaa);
            gradients.Add(dba);

            return gradients;
        }

        public List<NDArray> rnn_backward(NDArray da, Tuple<List<Cache>, NDArray> caches)
        {
            //var (caches, x) = caches;
            //var (a1, a0, x1, parameters) = caches[0];
            //var (Wax, Waa, Wya, ba, by) = parameters;
            var caches_ = caches.Item1;
            var x = caches.Item2;

            var a1 = caches_[0].a_next;
            var a0 = caches_[0].a_prev;
            var x1 = caches_[0].xt;

            var Wax = caches_[0].Wax;
            var Waa = caches_[0].Waa;
            var Wya = caches_[0].Wya;
            var ba = caches_[0].ba;
            var by = caches_[0].by;

            int n_a, m, T_x;
            n_a = da.Shape[0];
            m = da.Shape[1];
            T_x = da.Shape[2];

            int n_x = x.Shape[0];

            NDArray dx = np.zeros((n_x, m, T_x));
            NDArray dWax = np.zeros((n_a, n_x));
            NDArray dWaa = np.zeros((n_a, n_a));
            NDArray dba = np.zeros((n_a, 1));
            NDArray da0 = np.zeros((n_a, m));
            NDArray da_prevt = np.zeros((n_a, m));

            for (int t = T_x - 1; t >= 0; t--)
            {
                var gradients = rnn_cell_backward(da_prevt + slice(da, t), caches.Item1[t]);

                var dxt = gradients[0];
                da_prevt = gradients[1];
                var dWaxt = gradients[2];
                var dWaat = gradients[3];
                var dbat = gradients[4];

                replace(ref dx, dxt, t);
                dWax += dWaxt;
                dWaa += dWaat;
                dba += dbat;
            }

            da0 = da_prevt;

            List<NDArray> gradients2 = new List<NDArray>();
            gradients2.Add(dx);
            gradients2.Add(da0);
            gradients2.Add(dWax);
            gradients2.Add(dWaa);
            gradients2.Add(dba);

            return gradients2;
        }

        public List<NDArray> lstm_cell_backward(NDArray da_next, NDArray dc_next, Cache cache)
        {
            //var (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache;
            var a_next = cache.a_next;
            var c_next = cache.c_next;
            var a_prev = cache.a_prev;
            var c_prev = cache.c_prev;
            var ft = cache.ft;
            var it = cache.it;
            var cct = cache.cct;
            var ot = cache.ot;
            var xt = cache.xt;

            int n_x = xt.Shape[0];
            int m_x = xt.Shape[1];
            int n_a = a_next.Shape[0];
            int m_a = a_next.Shape[1];


            var dot = da_next * np.tanh(c_next) * ot * (1 - ot);
            var dcct = (dc_next * it + ot * (1 - np.power(np.tanh(c_next), 2)) * it * da_next) * (1 - np.power(cct, 2));
            var dit = (dc_next * cct + ot * (1 - np.power(np.tanh(c_next), 2)) * cct * da_next) * it * (1 - it);
            var dft = (dc_next * c_prev + ot * (1 - np.power(np.tanh(c_next), 2)) * c_prev * da_next) * ft * (1 - ft);

            var concat = np.concatenate((a_prev, xt), axis: 0);
            
            var dWf = np.dot(dft, concat.T);
            
            var dWi = np.dot(dit, concat.T);
            var dWc = np.dot(dcct, concat.T);
            var dWo = np.dot(dot, concat.T);

            var dbf = sum(dft, axis: 1, keepdims: true);
            var dbi = sum(dit, axis: 1, keepdims: true);
            var dbc = sum(dcct, axis: 1, keepdims: true);
            var dbo = sum(dot, axis: 1, keepdims: true);

            var da_prev = np.dot(slice(Wf, n_a, 1).T, dft) + np.dot(slice(Wi, n_a, 1).T, dit) + np.dot(slice(Wc, n_a, 1).T, dcct)
                + np.dot(slice(Wo, n_a, 1).T, dot);
            var dc_prev = dc_next * ft + ot * (1 - np.power(np.tanh(c_next), 2)) * ft * da_next;

            var dxt = np.dot(slice(Wf, n_a, false).T, dft) + np.dot(slice(Wi, n_a, false).T, dit) +
                np.dot(slice(Wc, n_a, false).T, dcct)
                + np.dot(slice(Wo, n_a, false).T, dot);

            List<NDArray> gradients = new List<NDArray>();
            gradients.Add(dxt);
            gradients.Add(da_prev);
            gradients.Add(dc_prev);
            gradients.Add(dWf);
            gradients.Add(dWi);
            gradients.Add(dWc);
            gradients.Add(dWo);
            gradients.Add(dbf);
            gradients.Add(dbi);
            gradients.Add(dbc);
            gradients.Add(dbo);

            return gradients;
        }

        public List<NDArray> lstm_backward(NDArray da, List<Cache> caches)
        {
            //var (caches, x) = caches;
            var caches_ = caches[0];
            //var x = caches.Item2; // ??

            var a1 = caches_.a_next;
            var a0 = caches_.a_prev;
            var x1 = caches_.xt;
            var Wf = caches_.Wf;
            var bf = caches_.bf;
            var Wi = caches_.Wi;
            var bi = caches_.bi;
            var Wo = caches_.Wo;
            var bo = caches_.bo;
            var Wc = caches_.Wc;
            var bc = caches_.bc;
            var Wy = caches_.Wy;
            var by = caches_.by;

            int n_a, m, T_x;
            n_a = da.Shape[0];
            m = da.Shape[1];
            T_x = da.Shape[2];

            int n_x = x1.Shape[0];
            m = x1.Shape[1]; // ??? double assignment

            NDArray dx = np.zeros((n_x, m, T_x));
            NDArray da0 = np.zeros((n_a, m));
            NDArray da_prevt = np.zeros(da0.Shape);
            NDArray dc_prevt = np.zeros(da0.Shape);
            NDArray dWf = np.zeros((n_a, n_a + n_x));
            NDArray dWi = np.zeros(dWf.Shape);
            NDArray dWc = np.zeros(dWf.Shape);
            NDArray dWo = np.zeros(dWf.Shape);
            NDArray dbf = np.zeros((n_a, 1));
            NDArray dbi = np.zeros(dbf.Shape);
            NDArray dbc = np.zeros(dbf.Shape);
            NDArray dbo = np.zeros(dbf.Shape);

            List<NDArray> gradients = new List<NDArray>();
            for (int t = T_x - 1; t >= 0; t--)
            {
                gradients = lstm_cell_backward(da_prevt + slice(da, t), dc_prevt, caches[t]);

                replace(ref dx, gradients[0], t);
                dWf += gradients[3];
                dWi += gradients[4];
                dWc += gradients[5];
                dWo += gradients[6];
                dbf += gradients[7];
                dbi += gradients[8];
                dbc += gradients[9];
                dbo += gradients[10];
            }
            da0 = gradients[1];

            var _gradients = new List<NDArray>();
            _gradients.Add(dx);
            _gradients.Add(da0);
            _gradients.Add(dWf);
            _gradients.Add(dWi);
            _gradients.Add(dWc);
            _gradients.Add(dWo);
            _gradients.Add(dbf);
            _gradients.Add(dbi);
            _gradients.Add(dbc);
            _gradients.Add(dbo);

            return _gradients;
        }
        private void print(string text)
        {
            Console.WriteLine(text);
        }

        private void replace(ref NDArray array, NDArray replace_with, int static_value)
        {
            for (int i = 0; i < array.Shape[0]; i++)
            {
                for (int j = 0; j < array.Shape[1]; j++)
                {
                    array[i, j, static_value] = replace_with[i, j];
                }
            }
        }

        private void _replace(ref NDArray array, NDArray replace_with, int max_value)
        {
            for (int i = 0; i < max_value; i++)
            {
                for (int j = 0; j < array.Shape[1]; j++)
                {
                    array[i, j] = replace_with[i, j];
                }
            }
        }
        
        private void replace(ref NDArray array, NDArray replace_with, int initial_value, bool anyValue)
        {
            for (int i = initial_value, _i = 0; i < array.Shape[0]; i++, _i++)
            {
                for (int j = 0; j < array.Shape[1]; j++)
                {
                    array[i, j] = replace_with[_i, j];
                }
            }
        }

        private NDArray slice(NDArray array, int static_index)
        {
            NDArray sliced = np.zeros((array.Shape[0], array.Shape[1]));
            for (int i = 0; i < array.Shape[0]; i++)
            {
                for (int j = 0; j < array.Shape[1]; j++)
                {
                    sliced[i, j] = array[i, j, static_index];
                }
            }

            return sliced;
        }

        private NDArray slice(NDArray array, int max_index, int anyValue)
        {
            NDArray sliced = np.zeros((array.Shape[0], max_index));
            for (int i = 0; i < array.Shape[0]; i++)
            {
                for (int j = 0; j < max_index; j++)
                {
                    sliced[i, j] = array[i, j];
                }
            }

            return sliced;
        }
        private NDArray slice(NDArray array, int max_index, bool t = false)
        {
            NDArray sliced = np.zeros((array.Shape[0], array.Shape[1]-max_index));
            for (int i = 0; i < array.Shape[0]; i++)
            {
                for (int j = max_index; j < array.Shape[1]; j++)
                {
                    sliced[i, j - max_index] = array[i, j];
                }
            }

            if (t)
                return sliced.T;
            
            return sliced;
        }

        private NDArray softmax(NDArray x)
        {
            NDArray e_x = np.exp(x - np.max(x));
            return e_x / sum(e_x);
        }

        private NDArray sigmoid(NDArray x)
        {
            int shape1 = x.Shape[0];
            int shape2 = x.Shape[1];
            NDArray result = np.zeros((shape1, shape2));
            for (int i = 0; i < shape1; i++)
            {
                for (int j = 0; j < shape2; j++)
                {
                    double value = x[i, j];
                    result[i, j] = 1 / (1 + Math.Exp(-value));
                }
            }
            return result;
        }

        private NDArray sum(NDArray x)
        {
            NDArray a = x[0];
            NDArray b = x[1];
            NDArray output = np.zeros(a.shape);
            for (int i = 0; i < a.size; i++)
            {
                output[i] = a[i] + b[i];
            }
            return output;
        }

        private double sum_single(NDArray x)
        {
            double sum = 0;
            for (int i = 0; i < x.size; i++)
            {
                sum += x[i];
            }
            return sum;

        }
        private NDArray sum(NDArray x, int axis, bool keepdims)
        {
            NDArray output = np.zeros((x.shape[0], axis));
            for (int i = 0; i < x.shape[0]; i++)
            {
                for (int j = 0; j < axis; j++)
                {
                    output[i, j] = sum_single(x[i]);
                }
            }
            return output;
        }
    }
}
