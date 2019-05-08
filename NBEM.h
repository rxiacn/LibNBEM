/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.20
* Author: Rui Xia
          rxia.cn@gmail.com
		  http://msrt.njust.edu.cn/staff/rxia
* Last updated on 2013-12-29
*********************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <math.h>

using namespace std;

struct sparse_feat
{
	vector<int> id_vec;
	vector<int> value_vec;
};

class NBEM
{
public:
	NBEM();
	~NBEM();

public:
	int class_set_size;
	int feat_set_size;

	vector<sparse_feat> samp_feat_vec;
    vector<int> samp_class_vec;
	vector<sparse_feat> usamp_feat_vec;
    vector< vector<float> > usamp_prb_vec;

	vector<int> samp_class_freq;
	vector<float> samp_class_prb;
	vector< vector<int> > samp_feat_class_freq;
    vector< vector<float> > samp_feat_class_prb;

	vector<float> usamp_class_freq;
	vector<float> usamp_class_prb;
    vector< vector<float> > usamp_feat_class_freq;
    vector< vector<float> > usamp_feat_class_prb;    

	vector<float> comb_class_prb;
    vector< vector<float> > comb_feat_class_prb;

private:
	vector<string> string_split(string terms_str, string spliting_tag);	
	void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);
	void load_train_data(string training_file);
	void load_unlabel_data(string unlabel_file);
	void load_samp_score(string score_file);

	void count_samp_class_freq();
	void calc_samp_class_prb(float cat_prior);
	void count_samp_feat_class_freq(); 
    void calc_samp_feat_class_prb(float token_cat_prior);
	void count_usamp_class_freq();
	void calc_usamp_class_prb(float cat_prior);
	void count_usamp_feat_class_freq();	
    void calc_usamp_feat_class_prb(float token_cat_prior);
    void calc_comb_class_prb(float lambda, float cat_prior);
    void calc_comb_feat_class_prb(float lambda, float token_cat_prior);

	vector<float> predict_logp(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm = 0);
	void predict_usamp_prb(vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm = 0);
	vector<float> score_to_prb(vector<float> &score);
	int score_to_class(vector<float> &score);

	double calc_samp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_usamp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_comb_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_logsum(vector<float> &logp_vec);
	
	float calc_acc(vector<int> &true_class_vec, vector<int> &pred_class_vec);
	void calc_prf(vector<int> &true_class_vec, vector<int> &pred_class_vec, map<int, vector<float> > &class_prf);

	//void alloc_uniform();

public:
	void save_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb);
    void load_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb);

    void learn_nb(string train_file, float cat_prior, float token_cat_prior);
	void learn_nbem_ssl(string train_file, string unlabel_file, int max_iter, double eps_thrd, float lambda, int len_norm, float cat_prior, float token_cat_prior, float init_token_cat_prior);
	void learn_nbem_usl(string init_file, string unlabel_file, int max_iter, double eps_thrd, int len_norm, float cat_prior, float token_cat_prior);

	float classify(string test_file, string output_file, int output_format, vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm);

};
