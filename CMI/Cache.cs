using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CMI
{
    public class Cache
    {
        public NDArray a_next;
        public NDArray a_prev;
        public NDArray xt;
        public NDArray Wax;
        public NDArray Waa;
        public NDArray Wya;
        public NDArray ba;
        public NDArray by;
        public NDArray c_next;
        public NDArray c_prev;
        public NDArray ft;
        public NDArray it;
        public NDArray cct;
        public NDArray ot;
        public List<NDArray> parameters = new List<NDArray>();
        public List<NDArray> parameters_lstm = new List<NDArray>();
        public NDArray[,] lstm_array;

        public NDArray Wf;
        public NDArray Wi;
        public NDArray Wc;
        public NDArray Wo;
        public NDArray Wy;
        public NDArray bf;
        public NDArray bi;
        public NDArray bo;
        public NDArray bc;
        private NDArray yt_pred;

        public Cache(NDArray a_next, NDArray a_prev, NDArray xt, List<NDArray> parameters)
        {
            this.a_next = a_next;
            this.a_prev = a_prev;
            this.parameters = parameters;
            this.xt = xt;
            Wax = parameters[0];
            Waa = parameters[1];
            Wya = parameters[2];
            ba = parameters[3];
            by = parameters[4];
        }

        public Cache(NDArray a_next, NDArray a_prev, NDArray xt, NDArray c_next, NDArray c_prev, NDArray ft, NDArray it, NDArray cct, NDArray ot, List<NDArray> parameters_lstm)
        {
            this.parameters_lstm = parameters_lstm;
            this.a_next = a_next;
            this.a_prev = a_prev;
            this.c_next = c_next;
            this.c_prev = c_prev;
            this.xt = xt;
            this.ft = ft;
            this.it = it;
            this.cct = cct;
            this.ot = ot;
            lstm_array = new NDArray[2, 4];
            lstm_array[0, 0] = ft;
            lstm_array[0, 1] = it;
            lstm_array[0, 2] = cct;
            lstm_array[0, 3] = ot;
            lstm_array[1, 0] = c_next;
            lstm_array[1, 1] = a_next;
            lstm_array[1, 2] = c_prev;
            lstm_array[1, 3] = a_prev;
        }

        public Cache(List<NDArray> parameters) // used for lstm_cell_backward
        {
            Wf = parameters[0];
            Wi = parameters[1];
            Wo = parameters[2];
            Wc = parameters[3];
            Wy = parameters[4];
            bf = parameters[5];
            bi = parameters[6];
            bo = parameters[7];
            bc = parameters[8];
            by = parameters[9];
        }

        public Cache(NDArray a_next, NDArray a_prev, NDArray xt, NDArray c_next, NDArray c_prev, NDArray ft, NDArray it, NDArray cct, NDArray ot, List<NDArray> parameters_lstm, NDArray yt_pred) : this(a_next, a_prev, xt, c_next, c_prev, ft, it, cct, ot, parameters_lstm)
        {
            this.a_next = a_next;
            this.a_prev = a_prev;
            this.xt = xt;
            this.c_next = c_next;
            this.c_prev = c_prev;
            this.ft = ft;
            this.it = it;
            this.cct = cct;
            this.ot = ot;
            this.parameters_lstm = parameters_lstm;
            this.yt_pred = yt_pred;
        }
    }
}
