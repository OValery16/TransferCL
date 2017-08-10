package com.example.myapplication;

import android.os.AsyncTask;
import android.view.View;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ScrollView;
import android.widget.TextView;

import com.transferCL.TransferCLlib;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    TransferCLlib t;
    String applicationName;
    TextView tv;
    ScrollView logContainer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        t=new TransferCLlib();


        applicationName = getApplicationContext().getPackageName();
        tv = (TextView)findViewById(R.id.textView4);

        try {
            Process process = Runtime.getRuntime().exec("logcat -c"); // we clear the logcat
        } catch (IOException e) {
            e.printStackTrace();
        }
        //////////////////

        tv = (TextView)findViewById(R.id.textView4);
        logContainer = (ScrollView)findViewById(R.id.SCROLLER_ID);

        new AsyncTask<Void, String, Void>() {

            @Override
            protected Void doInBackground(Void... params) {
                try {
                    Process process = Runtime.getRuntime().exec("logcat ActivityManager:I TransferCL:D *:S");
                    BufferedReader bufferedReader = new BufferedReader(
                            new InputStreamReader(process.getInputStream()));

                    String line = "";
                    while ((line = bufferedReader.readLine()) != null) {
                        line=line+"\n";
                        if (line.contains(":"))
                            line=line.substring(line.indexOf(":")+1);
                        publishProgress(line);
                    }
                }
                catch (IOException e) {
                }
                return null;
            }

            @Override
            protected void onProgressUpdate(String... values) {
                tv.append(values[0] + "\n");
                logContainer.post(new Runnable() {
                    @Override
                    public void run() {
                        logContainer.fullScroll(View.FOCUS_DOWN);
                    }
                });
            }
        }.execute();



    }

    public void prepareTrainingFiles(View v) {
        //this method prepares the training files (the training file and their labels are respectively stored in one binary file) and the mean And stdDev are stored in one file

        Runnable runnable = new Runnable() {
            public void run() {

                String fileNameStoreData="/data/data/"+applicationName+"/directoryTest/mem2Character2ManifestMapFileData2.raw";
                String fileNameStoreLabel= "/data/data/"+applicationName+"/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
                String fileNameStoreNormalization="/data/data/"+applicationName+"/directoryTest/normalizationTranfer.txt";

                t.prepareFiles("/data/data/"+applicationName+"/", fileNameStoreData,fileNameStoreLabel, fileNameStoreNormalization,"/sdcard1/character2/manifest6.txt",128, 1);

            }
        };
        Thread mythread = new Thread(runnable);
        mythread.start();



    }

    public void trainingModel(View v) {
        // this method trains our neural network at the native level

        Runnable runnable = new Runnable() {
            public void run() {


                String filename_label="/data/data/"+applicationName+"/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
                String filename_data="/data/data/"+applicationName+"/directoryTest/mem2Character2ManifestMapFileData2.raw";
                int imageSize=28;
                int numOfChannel=1;//black and white => 1; color =>3
                String storeweightsfile="/data/data/"+applicationName+"/directoryTest/weightstTransferedTEST.dat";
                String loadweightsfile="/sdcard1/preloadingData/weightstface1.dat";
                String loadnormalizationfile="/data/data/"+applicationName+"/directoryTest/normalizationTranfer.txt";
                String networkDefinition="1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n";// see https://github.com/hughperkins/DeepCL/blob/master/doc/Commandline.md
                int numepochs=100;
                int batchsize=128;
                int numtrain=128;
                float learningRate=0.01f;

                String cmdString="train filename_label="+filename_label;
                cmdString=cmdString+" filename_data="+filename_data;
                cmdString=cmdString+" imageSize="+Integer.toString(imageSize);
                cmdString=cmdString+" numPlanes="+Integer.toString(numOfChannel);
                cmdString=cmdString+" storeweightsfile="+storeweightsfile;
                cmdString=cmdString+" loadweightsfile="+loadweightsfile;
                cmdString=cmdString+" loadnormalizationfile="+loadnormalizationfile;
                cmdString=cmdString+" netdef="+networkDefinition;
                cmdString=cmdString+" numepochs="+Integer.toString(numepochs);
                cmdString=cmdString+" batchsize="+Integer.toString(batchsize);
                cmdString=cmdString+" numtrain="+Integer.toString(numtrain);
                cmdString=cmdString+" learningrate="+Float.toString(learningRate);

                String appDirctory ="/data/data/"+applicationName+"/";

                t.training(appDirctory,cmdString);
            }
        };
        Thread mythread = new Thread(runnable);
        mythread.start();


    }

    public void predictImages(View v) {
        // this method performs the prediction and sore the result in a file

        Runnable runnable = new Runnable() {
            public void run() {
                String appDirctory ="/data/data/"+applicationName+"/";
                String cmdString ="./predict weightsfile=/data/data/"+applicationName+"/directoryTest/weightstTransferedTEST.dat  inputfile=/sdcard1/character/manifest4.txt outputfile=/data/data/"+applicationName+"/preloadingData/pred2.txt";
                t.prediction(appDirctory,cmdString);

            }
        };
        Thread mythread = new Thread(runnable);
        mythread.start();

    }


    @Override
    public void onClick(View v) {

    }


}
