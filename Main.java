import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.NamedPlotColor;
import com.panayotis.gnuplot.style.PlotStyle;
import com.panayotis.gnuplot.style.Style;
import com.panayotis.gnuplot.swing.JPlot;

import java.io.*;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        JavaPlot jp = new JavaPlot("C:\\gnuplot\\bin\\gnuplot.exe");
        char[] buf = new char[1024];
        int inputlength = 4;
        int inputElementsInOneLine = 4;
        FileReader fw = new FileReader("transformation.txt");
        fw.read(buf);
        fw.close();
        String inputString = String.valueOf(buf);
        System.out.println(inputString);
        Scanner sc = new Scanner(inputString);
        double[][] inputs = new double[inputlength][4];
        for(int i=0;i<inputlength;i++)
        {
            for(int j=0;j<inputElementsInOneLine;j++)
            {
                inputs[i][j]=sc.nextDouble();
            }
        }
        int INPUTS = 4;
        int OUTPUTS= 4;
        int HIDDENLAYERNEURONS = 3;
        double LEARNINGRATE = 1;
        double MOMENTRATE = 0;
        boolean ISBIAS = true;

        Network network = new Network(INPUTS, OUTPUTS, HIDDENLAYERNEURONS,ISBIAS);
;
        LinkedList<Integer> indeksy = new LinkedList<>();
        for(int i=0;i<inputlength;i++) {
            indeksy.add(i);
        }
        //Kolejne epoki
        LinkedList<Double> errors = new LinkedList<>();
        int epochCounter = 1;
        double msefromEpoch;
        do{
            Collections.shuffle(indeksy);
            msefromEpoch=0;
            for(int i: indeksy)
            {
                network.generateOutputs(inputs[i],ISBIAS);
                network.backPropagation(inputs[i], ISBIAS);
                network.updateWeights(inputs[i], LEARNINGRATE, MOMENTRATE, ISBIAS);
               msefromEpoch+= network.getTotalMSE();
            }
            msefromEpoch/= OUTPUTS;
            errors.add(msefromEpoch);
            System.out.println("Epoka: "+ epochCounter + " MSE:   " + String.format("%.10f", msefromEpoch));

            epochCounter++;
        }while (epochCounter<1000 );
        double[][] plot = new double[epochCounter-1][2];
        for(int i=0;i<epochCounter-1;i++) {
            plot[i][0] = i;
            plot[i][1] = errors.get(i);
        }
        DataSetPlot ds = new DataSetPlot(plot);
        ds.setTitle("mse");
        PlotStyle style = new PlotStyle();
        style.setStyle(Style.POINTS);
     //   style.setLineType(7);
        style.setPointType(7);
        style.setLineType(NamedPlotColor.BLUE);

        ds.setPlotStyle(style);

        jp.addPlot(ds);
        jp.plot();

    }

}
