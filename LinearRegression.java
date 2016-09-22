import java.lang.Math;

public class LinearRegression{
    double[] features;
    double[] y;
    double[] yHat;
    double b0 = Math.random();
    double b1 = Math.random();
    double alpha;
    int epochs;
    public LinearRegression(double[] X, double[] y, double alpha, int epochs){
	this.features = X;
	this.y = y;
	this.alpha = alpha;
	this.epochs = epochs;
	this.yHat = new double[y.length];
	for(int i = 0; i < y.length; i++){
	    this.yHat[i] = 0;
	}
    }
    private double[] findYHat(){
	for(int i = 0; i < this.y.length; i++){
	    this.yHat[i] = this.b0 + this.b1 * features[i];
	}
	return yHat;
    }
    private double costFunction(){
	double totalSquaredError = 0;
	double error;
	this.yHat = findYHat();
	for(int i = 0; i < y.length; i++){
	    error = this.yHat[i] - this.y[i];
	    totalSquaredError += Math.pow(error , 2);
	}
	return totalSquaredError / y.length;
    }
    private void gradientDescent(double totalSquaredError){
	double b0_Gradient = 0;
	double b1_Gradient = 0;
	double squaredError;
	int randomInt = (int) Math.floor((double)Math.random()*features.length);
	squaredError = costFunction();
	for(int i = 0; i < y.length; i++){
	    b0_Gradient += yHat[i] - y[i];
	    b1_Gradient += (-features[i]) * (y[i] - yHat[i]);
	}
	b0_Gradient = b0_Gradient * (2 / y.length);
	b1_Gradient = b1_Gradient * (2 / y.length);
	this.b0 = b0 - this.alpha * b0_Gradient;
	this.b1 = b1 - this.alpha * b1_Gradient;
    }
    public double[] predict(){
	double squaredError;
	for(int i = 0; i < epochs; i++){
	    squaredError = costFunction();
	    gradientDescent(squaredError);
	    
	}
	return this.yHat;
    }
    public static void main(String[] args){
	double[] X = {1,2,4,3,5};
	double[] y = {1,3,3,2,5};
	double alphaTest = 0.001;
	int numOfEpochs = 100;
	LinearRegression newLinearReg = new LinearRegression(X,y, alphaTest, numOfEpochs);
	double[] hypothesis = newLinearReg.predict();
	for(int i = 0; i < hypothesis.length; i++){
	    System.out.println("Prediction: " + hypothesis[i]+ ", Actual: " + y[i]);
	}
    }
}
