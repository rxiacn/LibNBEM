/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.20
* Author: Rui Xia
          rxia.cn@gmail.com
		  http://msrt.njust.edu.cn/staff/rxia
* Last updated on 2013-12-29
*********************************************************************/

#include "NBEM.h"

NBEM::NBEM()
{
}

NBEM::~NBEM()
{
}

void NBEM::save_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    for (int j = 0; j < class_set_size; j++) {
        fout << samp_class_prb[j] << " ";
    }
    fout << endl;
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << samp_feat_class_prb[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void NBEM::load_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb)
{
    cout << "Loading model..." << endl;
    samp_class_prb.clear();
    samp_feat_class_prb.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
    }
    string line_str;
    // load class_prb
    getline(fin, line_str);
    vector<string> frist_line_vec = string_split(line_str, " ");
    for (vector<string>::iterator it = frist_line_vec.begin(); it != frist_line_vec.end(); it++) {
        float prb = (float)atof(it->c_str());
        samp_class_prb.push_back(prb);
        
    }
    // load feat_class_prb
    while (getline(fin, line_str)) {
        vector<float> prb_vec;
        vector<string> line_vec = string_split(line_str, " ");
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            float prb = (float)atof(it->c_str());
            prb_vec.push_back(prb);
        }
        samp_feat_class_prb.push_back(prb_vec);
    }
    fin.close();
    feat_set_size = (int)samp_feat_class_prb.size();
    class_set_size = (int)samp_feat_class_prb[0].size();
}

void NBEM::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
	int k = 0;
	string line_str;
    while (getline(fin, line_str)) {
		if (k == 0 && line_str[0] == '#'){
			vector<string> class_feat_size = string_split(line_str.substr(1), " ");
			class_set_size =  (int)atoi(class_feat_size[0].c_str());
			feat_set_size =  (int)atoi(class_feat_size[1].c_str());			
		} 
		else {
			size_t class_pos = line_str.find_first_of("\t");
			int class_id = atoi(line_str.substr(0, class_pos).c_str());
			samp_class_vec.push_back(class_id);
			string terms_str = line_str.substr(class_pos+1);
			sparse_feat samp_feat;
			if (terms_str != "") {
				vector<string> fv_vec = string_split(terms_str, " ");
				for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) {
					size_t feat_pos = it->find_first_of(":");
					int feat_id = atoi(it->substr(0, feat_pos).c_str());
					int feat_value = (int)atof(it->substr(feat_pos+1).c_str());
					if (feat_value != 0) {
						samp_feat.id_vec.push_back(feat_id);
						samp_feat.value_vec.push_back(feat_value);              
					}
				}
			}
			samp_feat_vec.push_back(samp_feat);		
		}
		k++;
    }
    fin.close();
}

void NBEM::load_train_data(string training_file)
{
    cout << "Loading training data..." << endl;
    read_samp_file(training_file, samp_feat_vec, samp_class_vec);
}

void NBEM::load_unlabel_data(string unlabel_file)
{
	cout << "Loading unlabeled data..." << endl;
	vector<int> usamp_class_vec; //meaningless
	read_samp_file(unlabel_file, usamp_feat_vec, usamp_class_vec);
}

void NBEM::learn_nb(string train_file,float cat_prior,float token_cat_prior)
{
    cout << "NB Learning..." << endl;
	load_train_data(train_file);
    count_samp_class_freq();
    calc_samp_class_prb(cat_prior);
	count_samp_feat_class_freq();
	calc_samp_feat_class_prb(token_cat_prior);
    samp_feat_vec.clear();
    samp_class_vec.clear();
}

void NBEM::learn_nbem_ssl(string train_file, string unlabel_file, int max_iter, double eps_thrd, float lambda, int len_norm, float cat_prior, float token_cat_prior, float init_token_cat_prior)
{
	// Model initial from labeled data
	cout << "\nModel initial training naive Bayes using labeled data..." << endl;
	load_train_data(train_file);
	count_samp_class_freq();
	calc_samp_class_prb(cat_prior);
	count_samp_feat_class_freq();
	// Set higher initial prior according to http://alias-i.com/lingpipe/demos/tutorial/em/read-me.html 
	// "We also found that having a more diffuse initial classifier (higher prior count) led to much better performance."
	calc_samp_feat_class_prb(init_token_cat_prior); 

	// evaluate on the test set (should be removed before release)
	string test_file = "test.samp";
	string output_file = "test.out";
	int output_format = 2;
	float acc_test = classify(test_file, output_file, output_format, samp_class_prb, samp_feat_class_prb, len_norm);
	cout << "Initial acc @ test set: " << acc_test << endl;	

	// 1st E-step
	cout << "\nEM for semi-supervised learning..." << endl;	
	double logl, logl_pre;
	load_unlabel_data(unlabel_file);
	predict_usamp_prb(samp_class_prb, samp_feat_class_prb, len_norm);
	logl = calc_comb_logl(samp_class_prb, samp_feat_class_prb);
	cout << "Initial loglikelihood @ labeled and unlabel training set: " << logl << endl;

	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_comb_class_prb(lambda, cat_prior);
		count_usamp_feat_class_freq();
		calc_comb_feat_class_prb(lambda, token_cat_prior);
		// E-step
		vector< vector<float> > usamp_prb_vec_pre = usamp_prb_vec;
		predict_usamp_prb(comb_class_prb, comb_feat_class_prb, len_norm);
		logl_pre = logl;
		logl = calc_comb_logl(comb_class_prb, comb_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << 100*(logl_pre-logl)/logl_pre  << "%" <<endl;
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
		// evaluate on the test set (should be removed before release)
		acc_test = classify(test_file, output_file, output_format, comb_class_prb, comb_feat_class_prb, len_norm);
		cout << "Acc @ test set: " << acc_test << endl;	
	}
	samp_feat_vec.clear();
	samp_class_vec.clear();
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}

void NBEM::learn_nbem_usl(string init_file, string unlabel_file, int max_iter, double eps_thrd, int len_norm, float cat_prior, float token_cat_prior)
{
	// Model initial from model file
	cout << "\nModel initial..." << endl;
	load_model(init_file, usamp_class_prb, usamp_feat_class_prb);

	// evaluate on the test set (should be removed before release)
	string test_file = "test.samp";
	string output_file = "test.out";
	int output_format = 2;
	float acc_test = classify(test_file, output_file, output_format, samp_class_prb, samp_feat_class_prb, len_norm);
	cout << "Initial acc @ test set: " << acc_test << endl;	

	// 1st E-step
	double logl, logl_pre;
	load_unlabel_data(unlabel_file);
	predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb, len_norm);
	logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
	cout << "\nEM learning..." << endl;	
	cout << "Initial loglikelihood: " << logl << endl;	
	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_usamp_class_prb(token_cat_prior);
		count_usamp_feat_class_freq();
		calc_usamp_feat_class_prb(token_cat_prior);
		// E-step
		predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb, len_norm);
		logl_pre = logl;
		logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << 100*(logl_pre-logl)/logl_pre  << "%" <<endl;
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
		// evaluate on the test set (should be removed before release)
		acc_test = classify(test_file, output_file, output_format, comb_class_prb, comb_feat_class_prb, len_norm);
		cout << "Acc @ test set: " << acc_test << endl;	
	}
	samp_feat_vec.clear();
	samp_class_vec.clear();
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}


void NBEM::count_samp_class_freq()
{
	// allocate
	samp_class_freq.clear();
	for (int j = 0; j < class_set_size; j++) {
		samp_class_freq.push_back(0);
	}
	// count freq
    for (vector<int>::iterator it_i = samp_class_vec.begin(); it_i != samp_class_vec.end(); it_i++) {
		int samp_class = *it_i;
        samp_class_freq[samp_class-1]++;
    }
 }
    
void NBEM::calc_samp_class_prb(float cat_prior)
{
	// allocate
	samp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		samp_class_prb.push_back(0.0);
	}
    // freq to prb
    for (int j = 0; j < class_set_size; j++) {
		samp_class_prb[j] = (float)(cat_prior+samp_class_freq[j])/(class_set_size*cat_prior+samp_class_vec.size());
    }
}

void NBEM::count_samp_feat_class_freq()
{
	// allocate
	samp_feat_class_freq.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<int> temp_vec1(class_set_size, 0);
		samp_feat_class_freq.push_back(temp_vec1);
	}
    // count freq
    for (size_t i = 0; i < samp_feat_vec.size(); i++) {
        sparse_feat samp_feat = samp_feat_vec[i];
        int samp_class = samp_class_vec[i];
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            samp_feat_class_freq[feat_id-1][samp_class-1] += feat_value;
        }
    }
}

void NBEM::calc_samp_feat_class_prb(float token_cat_prior)
{
	// allocate
	samp_feat_class_prb.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec2(class_set_size, 0.0);
		samp_feat_class_prb.push_back(temp_vec2);
	}	
    // column sum
    vector<int> samp_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            samp_feat_class_sum[j] += samp_feat_class_freq[k][j];
        }
    }
    // freq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
			// with Laplace smoothing
			samp_feat_class_prb[k][j] = (float)(token_cat_prior + samp_feat_class_freq[k][j])/(feat_set_size*token_cat_prior + samp_feat_class_sum[j]);
        }
    } 
}


void NBEM::count_usamp_class_freq()
{
	// allocate
	usamp_class_freq.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_freq.push_back(0.0);
	}
	// count expected freq
	for (size_t i = 0; i < usamp_prb_vec.size(); i++) {
		for (int j = 0; j < class_set_size; j++) {
			usamp_class_freq[j] += usamp_prb_vec[i][j];
		}
	}	
}
	
void NBEM::calc_usamp_class_prb(float cat_prior)
{
	// allocate
	usamp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb.push_back(0.0);
	}
	// freq to prb
    for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb[j] = (float)(cat_prior+usamp_class_freq[j])/(class_set_size*cat_prior + usamp_prb_vec.size());
    }
}

void NBEM::count_usamp_feat_class_freq()
{
	// allocate
	usamp_feat_class_freq.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		usamp_feat_class_freq.push_back(temp_vec);
	}
	// count expected freq
    for (size_t i = 0; i != usamp_feat_vec.size(); i++) {
        sparse_feat usamp_feat = usamp_feat_vec[i];
        vector<float> usamp_prb = usamp_prb_vec[i];
		for (int j = 0; j < class_set_size; j++) {
			for (size_t k = 0; k < usamp_feat.id_vec.size(); k++) {
				int feat_id = usamp_feat.id_vec[k];
				int feat_value = usamp_feat.value_vec[k];
				usamp_feat_class_freq[feat_id-1][j] += usamp_prb[j]*feat_value;
			}
		}
    }
}
    
void NBEM::calc_usamp_feat_class_prb(float token_cat_prior)
{
	// allocate
	usamp_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		usamp_feat_class_prb.push_back(temp_vec);
	}    
    // column sum
    vector<float> usamp_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            usamp_feat_class_sum[j] += usamp_feat_class_freq[k][j];
        }
    }
    // feaq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            usamp_feat_class_prb[k][j] = (float)(token_cat_prior + usamp_feat_class_freq[k][j]) / (feat_set_size + usamp_feat_class_sum[j]);
        }
    } 
}

void NBEM::calc_comb_class_prb(float lambda, float cat_prior)
{
	// allocate
	comb_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		comb_class_prb.push_back(0.0);
	}
	// comb prb
    for (int j = 0; j < class_set_size; j++) {
		comb_class_prb[j] = (cat_prior + samp_class_freq[j] + lambda*usamp_class_freq[j]) / (cat_prior * class_set_size + samp_class_vec.size() + lambda*usamp_prb_vec.size());
    }
}

void NBEM::calc_comb_feat_class_prb(float lambda, float token_cat_prior)
{
	// allocate
	vector< vector<float> > comb_feat_class_freq;
	comb_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		comb_feat_class_freq.push_back(temp_vec);
		comb_feat_class_prb.push_back(temp_vec);
	}
	// comb freq
	for (int k = 0; k < feat_set_size; k++) {
		for (int j = 0; j < class_set_size; j++) {
			comb_feat_class_freq[k][j] = samp_feat_class_freq[k][j] + lambda * usamp_feat_class_freq[k][j];
		}
	}
    // column sum
    vector<float> comb_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            comb_feat_class_sum[j] += comb_feat_class_freq[k][j];
        }
    }
    // freq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
			// with Laplace smoothing
			comb_feat_class_prb[k][j] = (float)(token_cat_prior + comb_feat_class_freq[k][j]) / (token_cat_prior * feat_set_size + comb_feat_class_sum[j]);
        }
    } 
}


vector<float> NBEM::predict_logp(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm)
{
    vector<float> logp(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        double logp_samp_given_class = 0.0;
        int samp_len = 0;
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            samp_len += feat_value;
            logp_samp_given_class += log(feat_class_prb[feat_id-1][j])*feat_value;
        }
        double logp_samp_and_class;
		// Lenght normalization according to http://lingpipe-blog.com/2009/02/13/document-length-normalized-naive-bayes/
		// and http://alias-i.com/lingpipe/demos/tutorial/em/read-me.html
        if (len_norm == 0) { // without length normalization
        	logp_samp_and_class = logp_samp_given_class + log(class_prb[j]);
		}
        else { // with length normalization
        	logp_samp_and_class = ((float)len_norm/samp_len) * logp_samp_given_class + log(class_prb[j]);
		}
        logp[j] = (float)logp_samp_and_class;
    }
    return logp;
}

void NBEM::predict_usamp_prb(vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm)
{
	usamp_prb_vec.clear();
	for (size_t i = 0; i != usamp_feat_vec.size(); i++) {
		vector<float> logp = predict_logp(usamp_feat_vec[i], class_prb, feat_class_prb, len_norm);
		vector<float> prb = score_to_prb(logp); // !!! check if length-norm could conduct on logp then prb
		usamp_prb_vec.push_back(prb);
	}
}

/*
vector<float> NBEM::score_to_prb(vector<float> &score)
{
	float m = score[0];
	for (int j = 1; j < class_set_size; j++) {
		if (score[j] > m) {
			m = score[j];	
		}
	}
    vector<float> prb(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        float denom = 0.0;
        if (score[j]-m > -10) {
			for (int i = 0; i < class_set_size; i++) {
				if (score[i]-m > t) {
					denom += exp(score[i]-m);			
				}
			}
			prb[j] = exp(score[j]-m)/denom;
        }
        else {
			prb[j] = 0;
        }
    }
    return prb;
}*/

vector<float> NBEM::score_to_prb(vector<float> &score)
// Compute without overflow: http://www.mblondel.org/journal/2010/06/21/semi-supervised-naive-bayes-in-python/
{
	float m = score[0];
	for (int j = 1; j < class_set_size; j++) {
		if (score[j] > m) {
			m = score[j];	
		}
	}
    vector<float> prb(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        float denom = 0.0;
        if (score[j]-m > -10) {
			for (int i = 0; i < class_set_size; i++) {
				if (score[i]-m > -10) {
					denom += exp(score[i]-m);		
				}
			}
			prb[j] = exp(score[j]-m)/denom;
        }
        else {
			prb[j] = 0;
        }
    }
    return prb;
}

int NBEM::score_to_class(vector<float> &score)
{
    int pred_class = 0; 
    float max_score = score[0];
    for (int j = 1; j < class_set_size; j++) {
        if (score[j] > max_score) {
            max_score = score[j];
            pred_class = j;
        }
    }
    return ++pred_class;
}

double NBEM::calc_comb_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double samp_logl = calc_samp_logl(class_prb, feat_class_prb);
	double usamp_logl = calc_usamp_logl(class_prb, feat_class_prb);
	return (samp_logl+usamp_logl);
}

double NBEM::calc_samp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double samp_logl;
	samp_logl = 0;
	for (size_t i = 0; i < samp_feat_vec.size(); i++) {
		sparse_feat samp_feat = samp_feat_vec[i];
		int samp_class = samp_class_vec[i];
		samp_logl += log(class_prb[samp_class-1]);
		for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
			//samp_logl += (len_norm/samp_feat.id_vec.size())*log(feat_class_prb[feat_id-1][samp_class-1])*feat_value; 
			samp_logl += log(feat_class_prb[feat_id-1][samp_class-1])*feat_value;
		}
	}
	return samp_logl;
}

double NBEM::calc_usamp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double usamp_logl = 0;
	for (size_t i = 0; i < usamp_feat_vec.size(); i++) {
		sparse_feat usamp_feat = usamp_feat_vec[i];
		vector<float> logp_vec = predict_logp(usamp_feat, class_prb, feat_class_prb);
		double logsum = calc_logsum(logp_vec); // maybe need some change, since lenght norm is conduct with log --> check tomorrow
		usamp_logl += logsum;
	}
	return usamp_logl;
}

double NBEM::calc_logsum(vector<float> &logp_vec)
// Compute log of sum of exp without overflow
// Ref: http://isites.harvard.edu/fs/docs/icb.topic540049.files/cs181_lec18_handout.pdf
{
	float max_logp = logp_vec[0];
	for (size_t j = 1; j < logp_vec.size(); j++) {
		if (logp_vec[j] > max_logp) {
			max_logp = logp_vec[j];
		}
	}
	double logsum = 0;
	double delta_sum = 0;
	for (size_t j = 0; j < logp_vec.size(); j++) {
		delta_sum += exp(logp_vec[j]-max_logp); //
	}
	logsum = log(delta_sum) + max_logp; 
	return logsum;
}

float NBEM::classify(string test_file, string output_file, int output_format, vector<float> &class_prb, vector< vector<float> > &feat_class_prb, int len_norm = 0)
{
    cout << "Classifying test file..." << endl;
    vector<sparse_feat> test_feat_vec;
    vector<int> test_class_vec;
    vector<int> pred_class_vec;
    read_samp_file(test_file, test_feat_vec, test_class_vec);
    ofstream fout(output_file.c_str());
    for (size_t i = 0; i < test_class_vec.size(); i++) {
        sparse_feat samp_feat = test_feat_vec[i];
        vector<float> pred_score;
		pred_score = predict_logp(samp_feat, class_prb, feat_class_prb, len_norm);
        int pred_class = score_to_class(pred_score);
        pred_class_vec.push_back(pred_class);
        fout << pred_class << "\t";
        if (output_format == 1) {
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_score[j] << ' '; 
            }       
        }
        else if (output_format == 2) {
            vector<float> pred_prb = score_to_prb(pred_score);
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_prb[j] << ' '; 
            }
        }
        fout << endl;       
    }
    fout.close();
	map<int, vector<float> > class_prf;
	calc_prf(test_class_vec, pred_class_vec, class_prf);
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc; 
}

void NBEM::calc_prf(vector<int> &test_class_vec, vector<int> &pred_class_vec, map<int, vector<float> > &class_prf)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
	set<int> class_set(test_class_vec.begin(), test_class_vec.end());
//	for (vector<int>::iterator it = test_class_vec.begin(); it != test_class_vec.end(); it++) {
//		class_set.insert(*it);
//	}
	map<int, vector<int> > class_count;
	for (set<int>::iterator itj = class_set.begin(); itj != class_set.end(); itj++) {
		vector<int> tmp_vec(4, 0);
		class_count.insert(make_pair(*itj, tmp_vec));
	}
    for (size_t id = 0; id != len; id++) {
		int test_class = test_class_vec[id];
		int pred_class = pred_class_vec[id];
		for (set<int>::iterator itj = class_set.begin(); itj != class_set.end(); itj++) {
			if (test_class == *itj && pred_class == *itj) { // true positive
				class_count[*itj][0] += 1;
			}
			else if (test_class == *itj && pred_class != *itj) { // false negative
				class_count[*itj][1] += 1;
			}
			else if (test_class != *itj && pred_class == *itj) { // false positive
				class_count[*itj][2] += 1;
			}
			else if (test_class != *itj && pred_class != *itj) { // true negative 
				class_count[*itj][3] += 1;			
			}
		}
	}
	class_prf.clear();
	for (set<int>::iterator itj = class_set.begin(); itj != class_set.end(); itj++) {
		float precision = (float)class_count[*itj][0] / (class_count[*itj][0] + class_count[*itj][2] + 1E-10);
		float recall = (float)class_count[*itj][0] / (class_count[*itj][0] + class_count[*itj][1] + 1E-10);
		float f_score = 2 * precision * recall / (precision + recall);
		vector<float> tmp_vec(3, 0.0);
		tmp_vec[0] = precision;
		tmp_vec[1] = recall;
		tmp_vec[2] = f_score;
		class_prf.insert(make_pair(*itj, tmp_vec));
		cout << "Class: " << *itj << "\tP: " << precision << "\tR: " << recall << "\tF: " << f_score << endl;
	}
}

float NBEM::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
    int err_num = 0;
    for (size_t id = 0; id != len; id++) {
        if (test_class_vec[id] != pred_class_vec[id]) {
            err_num++;
        }
    }
    float acc = 1 - ((float)err_num) / len;		
	cout << "Acc: " << acc << endl;	
	return acc;
}

vector<string> NBEM::string_split(string terms_str, string spliting_tag)
{
    vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) {
        if (term_end_pos > term_beg_pos) {
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
    if (term_beg_pos < terms_str.size()) {
        string end_str = terms_str.substr(term_beg_pos);
        feat_vec.push_back(end_str);
    }
    return feat_vec;
}

/*
void NBEM::alloc_uniform()
{
	usamp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb.push_back(1.0/class_set_size);
	}
	usamp_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 1.0/feat_set_size);
		usamp_feat_class_prb.push_back(temp_vec);
	}
}*/

