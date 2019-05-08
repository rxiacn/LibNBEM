/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.20
* Author: Rui Xia
          rxia.cn@gmail.com
		  http://msrt.njust.edu.cn/staff/rxia
* Last updated on 2013-12-29
*********************************************************************/

#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>
#include "NBEM.h"

using namespace std;


void print_help() {
	cout << "\n***** OpenPR-NB Classification Module *****\n\n"
		<< "usage: nb_classify [options] test_file model_file output_file\n\n"
		<< "options: -h        -> help\n"
		<< "         -l int    -> Length normalization factor (default 9.0)\n"
		<< "         -f [0..2] -> 0: only output class label (default)\n"
		<< "                   -> 1: output class label with log-likelihood\n"
		<< "                   -> 2: output class label with probability\n"

		<< endl;
}

void read_parameters(int argc, char *argv[], char *test_file, char *model_file, 
						char *output_file, int *output_format, int *len_norm) 
{
	// set default options
	*output_format = 0;
	*len_norm=9;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'f':
				*output_format = atoi(argv[++i]);
				break;
			case 'l':
				*len_norm = atoi(argv[++i]);
				break;
			default:
				cout << "Error: unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+2)>=argc) {
		cout << "Error: not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy(test_file, argv[i]);
	strcpy(model_file, argv[i+1]);
	strcpy(output_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
	char test_file[200];
	char model_file[200];
	char output_file[200];
	int output_format;
	int len_norm;
	read_parameters(argc, argv, test_file, model_file, output_file, &output_format, &len_norm);
    NBEM nbem;
	nbem.load_model(model_file, nbem.samp_class_prb, nbem.samp_feat_class_prb);
	float acc = nbem.classify(test_file, output_file, output_format, nbem.samp_class_prb, nbem.samp_feat_class_prb, len_norm);
	cout << "Accuracy: " << acc << endl;
	return 1;
}
