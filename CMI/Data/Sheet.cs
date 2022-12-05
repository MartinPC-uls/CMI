namespace CMI.Data
{
    public class Sheet
    {
        private string FilePath;
        public char[] Data;

        public Sheet(string filePath, int inputSize = 0)
        {
            FilePath = filePath;
            Read(inputSize);
        }

        private void Read(int inputSize)
        {
            Data = File.ReadAllText(FilePath).ToCharArray();
            if (inputSize > 0)
            {
                Data = Data.Skip(inputSize).ToArray();
            }
        }
    }
}
