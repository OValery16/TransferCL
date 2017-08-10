
#include "ConfigManager.h"
#include "../TransferCL/src/util/stringhelper.h"


using namespace std;


ConfigManager::ConfigManager(std::string fileDirectory){

	string filepath="";
	string line;
	ifstream myfileI (fileDirectory+"/binariesKernel/list.txt");
	if (myfileI.is_open())
	{
	   while ( getline (myfileI,line) )
	   {
		   vector<string> splitData=split(line, ",");
		   listOfCompiledKernel.insert ({splitData[0],splitData[1]});
        }
    myfileI.close();
    }
	kernellList=fileDirectory+"/binariesKernel/list.txt";
	binaryRepo=fileDirectory+"/binariesKernel/";



}




bool ConfigManager::alreadyCompiledKernel(string kernelname, string option,string operation,string &filepath){

	if (listOfCompiledKernel.empty()!=0){
		filepath=binaryRepo+operation+"_"+kernelname+".bin";
		return false;
	}
	string key=operation+" "+kernelname+" "+option;
	std::unordered_map<string,string>::const_iterator got = listOfCompiledKernel.find (key);

	if ( got == listOfCompiledKernel.end() ){

	    int i=0;
	    for (auto& x: listOfCompiledKernel) {
	    	if (x.first.find(kernelname)!=std::string::npos){
        		i++;
        	}
	    	string kernelname2=operation+"_"+kernelname+"_"+std::to_string(i);
	    	filepath=binaryRepo+kernelname2+".bin";
	      }
	    return false;
	}else{

		  filepath=got->second.c_str();
		  return true;
	  }

	return true;
}

void ConfigManager::addNewCompiledKernel(string kernelname, string options,string operation,string &filepath){

	string key=operation+" "+kernelname+" "+options;
	listOfCompiledKernel.insert ({key,filepath});

	ofstream myfileO;
	myfileO.open (kernellList,std::ofstream::out | std::ofstream::app);
	myfileO <<operation<<" "<<kernelname<<" " <<options <<","<<filepath<<"\n";
	myfileO.close();
}



