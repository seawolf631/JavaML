import java.lang.Math;

public class NaiveBayes{
    double[][] X;
    double[] y;
    double[] logPrior = new double[2];
    double[] logLikelihood = new double[2];
    double numOf0;
    double numOf1;
    double[][] test;
    double[] predictions;
    public NaiveBayes(double[][] X, double[] y){
	this.X = X;
	this.y = y;
	findNumOfEachClass();
	findLogPrior();
    }
    private void findNumOfEachClass(){
	for(int i = 0; i < y.length; i++){
	    if(y[i] == 0) numOf0++;
	    if(y[i] == 1) numOf1++;
	}
    }
    private void findLogPrior(){
	logPrior[0] = Math.log(numOf0 / (numOf0 + numOf1));
	logPrior[1] = Math.log(numOf1 / (numOf0 + numOf1));
    }
    private double findLogLikelihood(double value, int location){
	double total = 0;
	double denominator = 0;
	for(int i = 0; i < X.length; i++){
	    if(X[i][location] == value)
		total++;
	}
	if(location == 0) denominator = numOf0;
	else{ denominator = numOf1;}
	return total / denominator;
    }
    public void predict(double[][] test){
        this.test = test;
	predictions = new double[test.length];
	logLikelihood[0] = 0;
	logLikelihood[1] = 0;
     	for(int i = 0; i < test.length; i++){
	    for(int j = 0; j < test[i].length; j++){
		double temp = findLogLikelihood(test[i][j], j);
		logLikelihood[j] = logLikelihood[j] + temp;
		
	    }
	    if((logPrior[0] + logLikelihood[0]) > (logPrior[1] + logLikelihood[1]))
		predictions[i] = 0;
	    else
		predictions[i] = 1;
	    System.out.println("Log Probability of 0:  "+ (logPrior[0] + logLikelihood[0]) +"Log Probability of 1:  "+ (logPrior[1] + logLikelihood[1]));
	}
    }
    public static void main(String[] args){
	double[][] X = { {-3,7}, {1,5}, {1,2}, {-2,0}, {2,3}, {-4,0}, {-1,1}};
	double[] y = {0,0,0,0,1,0,1};
	NaiveBayes newNaiveBayes = new NaiveBayes(X,y);
	double[][] test = {{1,2}, {3,4}, {-1,1}};
	newNaiveBayes.predict(test);
	double[] results = newNaiveBayes.predictions;
	for(int i = 0; i < results.length; i++)
	    System.out.println(results[i]);
    }
}
