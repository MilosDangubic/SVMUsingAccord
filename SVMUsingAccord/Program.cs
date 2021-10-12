using System;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
namespace SVMUsingAccord
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======SVM CLASSIFICATION USING ACCORD ============");
            string fileName = "../../../winequality-red.txt";
            //Ucitavanje podataka
            Console.WriteLine("Loading data");
            double[][] data = Helper.MatrixLoad(fileName, true, ';');
            //Fisher-Yates algoritam za permutaciju observacija u skupu podataka. Menjamo raspored observacija u skupu
            Helper.randomize(data); 
            double[] allLabels = Helper.ExtractLabels(data);
            //U data setu ne postoje klase od 0 do 2 i Accord nam pravi problem, tako da umesto opsega klasa od 3 do 10
            //oduzimanjem od svage klase vrednost 3,dobijamo opseg klasa od 0 do 7
            Helper.labelModify(allLabels);
            data = Helper.RemoveLabels(data);
            int[] allLabelsI = Helper.ConvertDoubleToInt(allLabels);
            Console.WriteLine("Feature selection");
            data=FeatureSelection.selectFeatures(data, allLabelsI,3);
            //70% podataka koristimo za trening skup,30% podataka za test skup
            int n = data.Length;
            int nTrain = n / 100 * 70;
            double[][] trainData = new double[nTrain][];
            double[] trainLabels = new double[nTrain];
            double[][] testData = new double[n - nTrain][];
            double[] testLabels = new double[n - nTrain];
            Helper.SplitTrainTestData(data, allLabels, trainData, testData, trainLabels, testLabels, n);
            int[] trainLabelsI = Helper.ConvertDoubleToInt(trainLabels);
            int[] testLabelsI = Helper.ConvertDoubleToInt(testLabels);
            //Koristimo MulticlassSupportVectorLearning klasu iz Accord biblioteke zajedno sa SequentialMinimalOptimization klasom za treniranje 
            //modela
            MulticlassSupportVectorLearning<Circular> teacher = new MulticlassSupportVectorLearning<Circular>()
            {
                Learner = (param) => new SequentialMinimalOptimization<Circular>()
                {

                    Kernel = new Circular(1)
                }
            };

            //Treniranje modela
            var machine = teacher.Learn(trainData, trainLabelsI);
			//Vrsimo predikciju nad trening skupom i racunamo preciznost
            int[] predicted = machine.Decide(trainData);
            double preciznost = Helper.CalculateAccuracy(predicted, trainLabelsI);
            Console.WriteLine("Preciznost svm modela je {0} ", preciznost * 100);
			//Vrsimo predikciju nad test skupom i merimo preciznost
            int[] predictedTest = machine.Decide(testData);
            double preciznostTest = Helper.CalculateAccuracy(predictedTest, testLabelsI);
            Console.WriteLine("Preciznost svm modela na test skupu  je {0} ", preciznostTest * 100);

        }
    }
}
