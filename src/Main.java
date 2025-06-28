import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("dataset/student_data.csv");
        Instances data = source.getDataSet();

        // Set the class index (last column)
        data.setClassIndex(data.numAttributes() - 1);

        // Train Decision
        Classifier model = new J48();
        model.buildClassifier(data);

        for (int i = 0; i < data.numInstances(); i++) {
            double prediction = model.classifyInstance(data.instance(i));
            String predicted = data.classAttribute().value((int) prediction);
            String actual = data.instance(i).stringValue(data.classIndex());
            String name = data.instance(i).stringValue(0);

            System.out.println("Student " + name + ": Actual = " + actual + ", Predicted = " + predicted);
        }
    }
}
