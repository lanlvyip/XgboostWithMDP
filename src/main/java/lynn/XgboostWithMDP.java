package lynn;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class XgboostWithMDP {

    public static void arrangeData(String path, String newPath) throws IOException {
        String contencts = new String(Files.readAllBytes(
                Paths.get(path)), StandardCharsets.UTF_8
        );
        List<String> lines = Arrays.asList(contencts.split("\r\n"));
        List<String> newLines = new ArrayList<>();
        lines.forEach(a -> {
            a = a.replace(",",":1 ");
            if(a.endsWith("N")){
                a = "0 " + a;
            }else{
                a = "1 " + a;
            }
            a = a.substring(0, a.length() - 1);
            newLines.add(a);
        });
        File file = new File(newPath);
        if(!file.exists()){
            file.createNewFile();
        }
        Files.write(Paths.get(newPath), newLines, StandardCharsets.UTF_8, StandardOpenOption.CREATE);


    }

    public static void main(String[] args) throws XGBoostError, IOException {
        arrangeData("src/main/resources/D1/train.txt","src/main/resources/D1/train1.txt");
        arrangeData("src/main/resources/D2/test.txt","src/main/resources/D2/test1.txt");

        DMatrix trainMat = new DMatrix("src/main/resources/D1/train1.txt");
        DMatrix testMat = new DMatrix("src/main/resources/D2/test1.txt");

        HashMap<String, Object> params = new HashMap<>();

        params.put("eta",1.0);
        params.put("max_depth",2);
        params.put("slient",1);
        params.put("objective","binary:logistic");

        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("train", trainMat);
        watches.put("test", testMat);

        int round = 3;
        Booster booster = XGBoost.train(trainMat, params, round, watches, null, null);

        float[][] predicts1 = booster.predict(testMat, false, 1);
        float[][] predicts2 = booster.predict(testMat);

        CustomEval eval = new CustomEval();
        System.out.println("error of predicts1: "  + eval.eval(predicts1, testMat));
        System.out.println("error of predicts2: "  + eval.eval(predicts2, testMat));

    }


}
