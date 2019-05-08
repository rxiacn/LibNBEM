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
	cout << "\n***** OpenPR-NBEM Un-supervised Learning Module *****\n\n"
		<< "usage: nbem [options] initial_model_file unlabeled_file model_file\n\n"
		<< "options: -h        -> help\n"
		<< "         -n int    -> Maximal iteration steps (default: 20)\n" 
		<< "         -m float  -> Minimal increase rate of loglikelihood (default: 1e-4)\n" 
		<< "         -l int    -> Length normalization factor (default 0)\n"
		<< "         -c float  -> class prior (default 0.01)\n"
		<< "         -t float  -> class-conditinal feature(token) prior (default 0.01)\n"
		<< endl;
}

void read_parameters(int argc, char *argv[], char *init_file, char *unlabel_file, char *model_file,
						int *max_iter, double *eps_thrd, int *len_norm, float *cat_prior, 
						float *token_cat_prior) 
{
	// set default options
	*max_iter = 20;
	*eps_thrd = 1e-4;
	*len_norm = 0;
	*cat_prior = 0.01;
	*token_cat_prior = 0.01;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'n':
				*max_iter = atoi(argv[++i]);
				break;
			case 'm':
				*eps_thrd = atof(argv[++i]);
				break;
			case 'l':
				*len_norm = atoi(argv[++i]);
				break;
			case 'c':
				*cat_prior = atof(argv[++i]);
				break;
			case 't':
				*token_cat_prior = atof(argv[++i]);
				break;
			default:
				cout << "Unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+2)>=argc) {
		cout << "Not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy(init_file, argv[i]);
	strcpy(unlabel_file, argv[i+1]);
	strcpy(model_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
	char init_file[200];
	char unlabel_file[200];
	char model_file[200];
	int max_iter;
	double eps_thrd;
	int len_norm;
	float cat_prior;
	float token_cat_prior;
	read_parameters(argc, argv, init_file, unlabel_file, model_file, &max_iter, &eps_thrd, &len_norm, &cat_prior, &token_cat_prior);
    NBEM nbem;
    nbem.learn_nbem_usl(init_file, unlabel_file, max_iter, eps_thrd, len_norm, cat_prior, token_cat_prior);
	nbem.save_model(model_file, nbem.usamp_class_prb, nbem.usamp_feat_class_prb);
	return 1;
}
