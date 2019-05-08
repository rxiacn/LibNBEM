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
	cout << "\n***** OpenPR-NBEM Supervised Learning (Naive Bayes) Module *****\n\n"
		<< "usage: nb_learn [options] training_file model_file\n\n"
		<< "options: -h        -> help\n"
		<< endl;
}

void read_parameters(int argc, char *argv[], char *training_file, char *model_file) 
{
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			default:
				cout << "Error: unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+1)>=argc) {
		cout << "Error: not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy (training_file, argv[i]);
	strcpy (model_file, argv[i+1]);
}

int main(int argc, char *argv[])
{
	char training_file[200];
	char model_file[200];
	read_parameters(argc, argv, training_file, model_file);
    NBEM nb;
    nb.learn_nb(training_file);
	nb.save_model(model_file, nb.samp_class_prb, nb.samp_feat_class_prb);
	return 1;
}
