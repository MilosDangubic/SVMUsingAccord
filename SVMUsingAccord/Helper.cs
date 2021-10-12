using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Text;

namespace SVMUsingAccord
{
    public class Helper
    {
        public static void labelModify(double[] labels) 
        {
            for(int i=0; i<labels.Length; i++) 
            {
                labels[i] -= 3;
            }
        }

        public static List<string> extractFeatureNames(string file, char sep)
        {
            List<string> featureNames = new List<string>();
            System.IO.FileStream ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr = new System.IO.StreamReader(ifs);
            string line = sr.ReadLine();
            string[] lineParts = line.Split(sep);

            for (int i = 0; i < lineParts.Length - 1; i++)
            {
                featureNames.Add(lineParts[i]);
            }

            return featureNames;

        }


        public static void randomize(double[][] data)
        {
            int n;
            n = data.Length;
            Random r = new Random();
            for (int i = n - 1; i > 0; i--)
            {
                int j = r.Next(0, i + 1);
                for (int k = 0; k < data[i].Length; k++)
                {
                    double temp = data[i][k];
                    data[i][k] = data[j][k];
                    data[j][k] = temp;
                }
            }
        }
        


        private static double[][] MatrixCreate(int rows, int cols)
        {
           
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public static double[] ExtractLabels(double[][] data)
        {
            double[] labels = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                labels[i] = data[i][data[i].Length - 1];
            }
            return labels;

        }

        public static double[][] RemoveLabels(double[][] datawithLabel)
        {

            double[][] data = new double[datawithLabel.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[datawithLabel[i].Length - 1];
            }

            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = datawithLabel[i][j];
                }
            }
            return data;
        }
      
        public static double[][] MatrixLoad(string file, bool header, char sep)
        {
            string line = "";
            string[] tokens = null;
            int ct = 0;
            int rows, cols;
            System.IO.FileStream ifs =
              new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr =
              new System.IO.StreamReader(ifs);
            while ((line = sr.ReadLine()) != null)
            {
                ++ct;
                tokens = line.Split(sep);
            }
            sr.Close(); ifs.Close();
            if (header == true)
                rows = ct - 1;
            else
                rows = ct;
            cols = tokens.Length;
            double[][] result = MatrixCreate(rows, cols);


            int i = 0; // row index
            ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
            sr = new System.IO.StreamReader(ifs);

            if (header == true)
                line = sr.ReadLine();  // consume header
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(sep);
                for (int j = 0; j < cols; ++j)
                    result[i][j] = double.Parse(tokens[j], System.Globalization.CultureInfo.InvariantCulture);
                ++i; // next row
            }
            sr.Close(); ifs.Close();
            return result;
        }

     
        public static int[] ConvertDoubleToInt(double[] array)
        {
            int[] newArray = new int[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                newArray[i] = (int)array[i];
            }
            return newArray;
        }

        public static void SplitTrainTestData(double[][] data, double[] allLabels, double[][] trainData, double[][] testData, double[] trainLabels, double[] testLabels, int n)
        {
            int nTrain = n / 100 * 70;
            for (int i = 0; i < nTrain; i++)
            {
                trainLabels[i] = allLabels[i];
                trainData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    trainData[i][j] = data[i][j];
                }
            }
            for (int i = 0; i < n - nTrain; i++)
            {
                testLabels[i] = allLabels[i + nTrain];
                testData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    testData[i][j] = data[i + nTrain][j];
                }
            }

        }


        
        public static double EvaluateModel(double[][] X, int[] Y)
        {
            MulticlassSupportVectorLearning<Circular> teacher = new MulticlassSupportVectorLearning<Circular>()
            {
                Learner = (param) => new SequentialMinimalOptimization<Circular>()
                {

                    Kernel = new Circular(1)
                }
            };

            //Treniranje modela
            var machine = teacher.Learn(X, Y);
            double acc = 0;
            int[] decisions = machine.Decide(X);
            acc = CalculateAccuracy(decisions, Y);
            return acc;
            
        }


        public static double CalculateAccuracy(int[] predicted, int[] actual)
        {
            double acc = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                if (predicted[i] == actual[i])
                {
                    acc++;
                }
            }
            acc = acc / predicted.Length;
            return acc;
        }




    }
}
