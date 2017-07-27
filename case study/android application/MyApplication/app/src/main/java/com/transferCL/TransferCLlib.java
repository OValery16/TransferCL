package com.transferCL;


import java.nio.charset.StandardCharsets;

public class TransferCLlib {
	
	static boolean sfoundLibrary = true;  
	static {
		  try {
              // load the native library
			  System.loadLibrary("transferCLNative");  
		  }
		  catch (UnsatisfiedLinkError e) {
		      sfoundLibrary = false;
		  }
		}
	// declare the native method
	public static native int training(String path, String cmdTrain);
	public static native int prediction(String path, String cmdPrediction);
	public static native int prepareFiles(String path, String fileNameStoreData,String fileNameStoreLabel, String fileNameStoreNormalization, String manifestPath, int nbImage, int imagesChannelNb);

}
