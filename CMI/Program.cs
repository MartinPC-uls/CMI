using CMI.Network;
//using CMI.Network2;
using static CMI.Data.Normalizer;
using static CMI.Utils;
using static CMI.Matrix;
using CMI.Data;
using MusicXML_Parser;
using System.Xml;
using MusicXML_Parser.Music;
using NumSharp;
using CMI;

List<char[]> input_chunks = new();
List<char[]> output_chunks = new();
void Split(char[] input, char[] output, int chunkSize = 0)
{
    if (chunkSize == 0)
        chunkSize = input.Length;
    if (chunkSize > input.Length)
        throw new ArgumentException("Chunk size is greater than input length.");
    for (int i = 0; i < input.Length; i++)
    {
        char[] input_chunk = new char[chunkSize];
        char[] output_chunk = new char[chunkSize];
        if (i + chunkSize > input.Length)
            break;
        for (int j = 0; j < chunkSize; j++)
        {
            input_chunk[j] = input[i + j];
            output_chunk[j] = output[i + j];
        }
        input_chunks.Add(input_chunk);
        output_chunks.Add(output_chunk);
    } 
}

Split(" LQS QULQXPSLPXEDNQIN QVXBVZQV]SXQUUJISVNSUVUSQGJGNEINNPQSEJEEPSSUVX@GLLVUL9@EELNJLNLP@EG>E@EI@ELDG@DL98BE=BDEEJL6EJNNQ".ToCharArray(), "LQS QULQXPSLPXEDNQIN QVXBVZQV]SXQUUJISVNSUVUSQGJGNEINNPQSEJEEPSSUVX@GLLVUL9@EELNJLNLP@EG>E@EI@ELDG@DL98BE=BDEEJL6EJNNQG".ToCharArray(), 10);

/*foreach (var input in input_chunks)
{
    foreach (var i in input)
    {
        Print(i, false);
    }
    Print("\n");
}*/

//LSTM lstm = new(3);
//LSTM lstm = new();
//lstm.LoadParameters();
//lstm.InitializeWeights();
//lstm.initialize();

MainProgram program = new();
/*

//lstm.SeqTrain(input_chunks, output_chunks, 10000000, 100000000);
//char[] input = "^ `^][` `^][[ YWWYV V JV JRVJVJRVVbV^b3:C3:C3".ToCharArray();
//char[] input =  " LQS QULQXPSLPXEDNQIN QVXBVZQV]SXQUUJISVNSUVUSQGJGNEINNPQSEJEEPSSUVX@GLLVUL9@EELNJLNLP@EG>E@EI@ELDG@DL98BE=BDEEJL6EJNNQ".ToCharArray();
char[] input =  "TT<".ToCharArray();
double[] _input = Normalize(input);
char[] output = "T<H".ToCharArray();
double[] _output = Normalize(output);

//double[] _input = new double[]  { 0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
//double[] _output = new double[] { 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

//double[] _input = new double[]  { 0.1, 0.3, 0.5, 0.7 };
//double[] _output = new double[] { 0.3, 0.5, 0.7, 0.9 };

//lstm.SeqPrediction(_input, 1);

//lstm.Train(_input, _output, 10000000, 1000000);
//lstm.SeqTrain(input_chunks, output_chunks, 100000, 100000);
//lstm.Train(_input, _output, 1000000, 100000000);

//lstm.SeqPrediction(_input, 10);
lstm.Prediction(_input);

SheetConfiguration music = new SheetConfiguration()
{
    WorkTitle = "Test",
    Composer = "Martín Pizarro",
    Accidental = true,
    Beam = true,
    Print_NewPage = true,
    Print_NewSystem = true,
    Stem = true,
    WordFont = "FreeSerif",
    WordFontSize = 10,
    LyricFont = "FreeSerif",
    LyricFontSize = 11,
    CreditWords_Title = "Just a test",
    CreditWords_Subtitle = "Testing",
    CreditWords_Composer = "Martín",
    StaffLayout = 2,
    StaffDistance = 65,
    Divisions = 24,
    Tempo = 134,
    TimeBeats = 4,
    TimeBeatType = 4
};
XmlDocument score = music.Init("asd.xml", Key.CMajor);
//XmlDocument score = sheet.Load(AppDomain.CurrentDomain.BaseDirectory + "score.xml");
XmlNode node = score.SelectSingleNode("score-partwise/part[@id='P1']");
XmlNode measure = music.AddMeasure(node);

foreach (var predicted in lstm.Predicted_Output)
{
    music.Add(predicted, 1, measure);
}

/*print("Predicted Output: ", false);
foreach (var predicted in lstm.Predicted_Output)
{
    print(Denormalize(predicted), false);
}
print("\n");

/*

// USING LSTM 1 WITH NO MATRICES
LSTM lstm = new();
//lstm.initialize();
lstm.loadParameters();
Sheet sheet = new(AppDomain.CurrentDomain.BaseDirectory + "dataset.txt");

char[] input  = "abcdefghi".ToCharArray();
char[] output = "ghijkl".ToCharArray();

double[] _input = Normalize(input);
double[] _output = Normalize(output);

//lstm.Train(_input, _output, 1000000, 100000);

lstm.SeqPrediction(_input, 10);
//lstm.Prediction(_input, 10);

SheetConfiguration music = new SheetConfiguration()
{
    WorkTitle = "Test",
    Composer = "Martín Pizarro",
    Accidental = true,
    Beam = true,
    Print_NewPage = true,
    Print_NewSystem = true,
    Stem = true,
    WordFont = "FreeSerif",
    WordFontSize = 10,
    LyricFont = "FreeSerif",
    LyricFontSize = 11,
    CreditWords_Title = "Just a test",
    CreditWords_Subtitle = "Testing",
    CreditWords_Composer = "Martín",
    StaffLayout = 2,
    StaffDistance = 65,
    Divisions = 24,
    Tempo = 134,
    TimeBeats = 4,
    TimeBeatType = 4
};
XmlDocument score = music.Init("score.xml", Key.FMajor);
//XmlDocument score = sheet.Load(AppDomain.CurrentDomain.BaseDirectory + "score.xml");
XmlNode node = score.SelectSingleNode("score-partwise/part[@id='P1']");
XmlNode measure = music.AddMeasure(node);

foreach (var predicted in lstm.Predicted_Output)
{
    print("PREDICTED: " + predicted);
    music.Add(predicted, 1, measure);
}


// USING LSTM 2 WITH MATRICES
/*CMI.Network2.LSTM lstm = new();
lstm.initialize();

char[] input = "ABCD".ToCharArray();
char[] output = "BCDE".ToCharArray();

double[] _input = Normalize(input);
double[] _output = Normalize(output);

// create an array of 10 double
double[] array = new double[10];
// fill the array with increasing values
for (int i = 0; i < array.Length; i++)
{
    array[i] = i;
    print(array[i]);
}

lstm.Train(_input, array, 100000000, 10);*/

/*
SheetConfiguration music = new SheetConfiguration()
{
    WorkTitle = "Test",
    Composer = "Martín Pizarro",
    Accidental = true,
    Beam = true,
    Print_NewPage = true,
    Print_NewSystem = true,
    Stem = true,
    WordFont = "FreeSerif",
    WordFontSize = 10,
    LyricFont = "FreeSerif",
    LyricFontSize = 11,
    CreditWords_Title = "Just a test",
    CreditWords_Subtitle = "Testing",
    CreditWords_Composer = "Martín",
    StaffLayout = 2,
    StaffDistance = 65,
    Divisions = 24,
    Tempo = 134,
    TimeBeats = 4,
    TimeBeatType = 4
};
XmlDocument score = music.Init("score.xml", Key.FMajor);
//XmlDocument score = sheet.Load(AppDomain.CurrentDomain.BaseDirectory + "score.xml");
XmlNode node = score.SelectSingleNode("score-partwise/part[@id='P1']");
XmlNode measure = music.AddMeasure(node);

foreach (var predicted in lstm.Predicted_Output)
{
    music.Add(predicted, 1, measure);
}*/