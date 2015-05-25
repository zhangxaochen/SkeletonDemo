#include "bodyPartRecognizer.h"

BPRTree::BPRTree(){
	_recognizer = 0;
	_value_mat = 0;
	_response_mat = 0;
	_var_type_mat = 0;
	data = 0;

	_nodeCount = 0;
	_featureCount = 0;
}

BPRTree::~BPRTree(){
	clear();
}

void BPRTree::clear(){
	CvDTree::clear();
}

bool BPRTree::train(BPRecognizer* recognizer)
{
	_recognizer = recognizer;
	_value_mat = _recognizer->_value_mat;
	_response_mat = _recognizer->_response_mat;
	_var_type_mat = _recognizer->_var_type_mat;
	_active_var_mask = _recognizer->_active_var_mask;

	clear();

	BPRecogPara& para = _recognizer->_para;

	// build sample set
	vector<DepthSample*>& trainData = _recognizer->_trainSamples->getSampleSet();
	if(trainData.empty()){
		ERROR_MSG("Empty training data set, please load", _recognizer->_moduleID);
		return false;
	}
	// random image
	MSG("Random images...");
	_rImg.clear();
	vector<bool> imgUse;
	imgUse.resize(trainData.size());
	RNG* rng = _recognizer->_rng;
	int imgNum = _recognizer->_para.img_per_tree;
	fill(imgUse.begin(), imgUse.end(), false);
	while(_rImg.size() < imgNum){
		int imgId = (*rng)((int)trainData.size());
		if(!imgUse[imgId]){
			_rImg.push_back(imgId);
			imgUse[imgId] = true;
		}
	}
	imgUse.clear();

	// init active mask
	CvMat submask1, submask2;
	cvGetCols(_active_var_mask, &submask1, 0, para.node_active_var);
	cvSet(&submask1, cvScalar(1));
	if(para.node_active_var < _active_var_mask->cols){
		cvGetCols(_active_var_mask, &submask2, para.node_active_var, _active_var_mask->cols);
		cvZero(&submask2);
	}

	// build value matrix
	MSG("Re-build value matrix...");
	rebuildValueMat();

	// set train data
	data = new CvDTreeTrainData();
	data->set_data(_value_mat, CV_ROW_SAMPLE, _response_mat, 0, 0,
		_var_type_mat, 0, _recognizer->_dPara);

	MSG("Grow tree...");
	bool res = do_train(0);

	// clear features
	MSG("Release features...");
	for(vector<pair<DepthFeature*, bool> >::iterator it=_curFeatures.begin();
		it!=_curFeatures.end(); it++){
			if(it->second){
				_featureCount++;
				if(para.feature_type == DEPTH_FEATURE_GSUB)
					_recognizer->addNewBestFeature((GSubDepthFeature*)it->first);
				else if(para.feature_type == DEPTH_FEATURE_GADD)
					_recognizer->addNewBestFeature((GAddDepthFeature*)it->first);
				else if(para.feature_type == DEPTH_FEATURE_GHALF)
					_recognizer->addNewBestFeature((GHalfDepthFeature*)it->first);
			}else{
				_recognizer->addNewNullFeature();
			}
			delete it->first;
	}
	_curFeatures.clear();

	return res;
}

void BPRTree::read(CvFileStorage* fs, CvFileNode* node, BPRecognizer* recognizer)
{
	_recognizer = recognizer;
	CvDTree::read(fs, node);
}

void BPRTree::rebuildValueMat()
{
	// random theta
	list<pair<CvPoint, CvPoint> > rTheta;
	BPRecogPara& para = _recognizer->_para;
	RNG* rng = _recognizer->_rng;
	while(rTheta.size() <_recognizer->_para.theta_per_tree){
		int u_x = (*rng).uniform(-para.feature_high_bound, para.feature_high_bound);
		int u_y = (*rng).uniform(-para.feature_high_bound, para.feature_high_bound);
		int v_x = (*rng).uniform(-para.feature_high_bound, para.feature_high_bound);
		int v_y = (*rng).uniform(-para.feature_high_bound, para.feature_high_bound);
		if(abs(u_x)<=3 && abs(u_y)<=3 && abs(v_x)<=3 && abs(v_y)<=3) continue;
		if(abs(u_x - v_x)<=3 && abs(u_y - v_y)<=3) continue;
		rTheta.push_back(make_pair(cvPoint(u_x, u_y), cvPoint(v_x, v_y)));
	}
	// random t if threshold feature
	if(para.feature_type == DEPTH_FEATURE_GSUB){
		for(list<pair<CvPoint, CvPoint> >::iterator it=rTheta.begin(); it!=rTheta.end(); it++){
			GSubDepthFeature* f = new GSubDepthFeature(it->first, it->second);
			_curFeatures.push_back(make_pair(f, false));
		}
	}else if(para.feature_type == DEPTH_FEATURE_GADD){
		for(list<pair<CvPoint, CvPoint> >::iterator it=rTheta.begin(); it!=rTheta.end(); it++){
			//double slop = (double)it->first.y / it->first.x;
			//double y = it->second.x * slop;
			//if(abs(y) <= abs(it->second.y)) it->second.y = y;
			//else it->second.x = it->second.y / slop;
			GAddDepthFeature* f = new GAddDepthFeature(it->first, it->second);
			_curFeatures.push_back(make_pair(f, false));
		}
	}
	else if(para.feature_type == DEPTH_FEATURE_GHALF){
		for(list<pair<CvPoint, CvPoint> >::iterator it=rTheta.begin(); it!=rTheta.end(); it++){
			GHalfDepthFeature* f = new GHalfDepthFeature(it->first, it->second);
			_curFeatures.push_back(make_pair(f, false));
		}
	}
	rTheta.clear();
	rTheta.resize(0);

	// update value matrix
	vector<DepthSample*>& trainData = _recognizer->_trainSamples->getSampleSet();
	float* value_mat_ptr = _value_mat->data.fl;
	int* response = _response_mat->data.i;
	LogCat::getInstancePtr()->addProcessBar("Update Value Matrix:", 50);
	double p_add = 1.0/_rImg.size();
	double p = p_add;
	for(list<int>::iterator it=_rImg.begin(); it!=_rImg.end(); it++, p += p_add){
		DepthSample* img = trainData[*it];
		// load image and random pixel
		int pixelNum = _recognizer->_para.pixel_per_img;
		img->loadImage();
		img->randomTrainPixel(pixelNum);
		list<CvPoint>& pixels = img->getTrainPixel();
		IplImage* depthImg = img->getDepthImg();
		IplImage* labelImg = img->getlabelImg();
		// update value
		for(list<CvPoint>::iterator itt=pixels.begin(); itt!=pixels.end(); itt++){
			CvPoint& p = *itt;
			for(vector<pair<DepthFeature*, bool> >::iterator fit=_curFeatures.begin(); fit!=_curFeatures.end(); fit++){
				float v = fit->first->getValue(depthImg, p);
				*value_mat_ptr = v;
				value_mat_ptr++;
			}
			*response = CV_IMAGE_ELEM(labelImg, BodyLabel, itt->y, itt->x);
			response++;
		}
		// release image and clear random pixel set
		img->releaseImage();
		img->clearTrainPixel();
		LogCat::getInstancePtr()->setProcessBar(p);
	}
	LogCat::getInstancePtr()->removeProcessBar();
}

CvDTreeSplit* BPRTree::find_best_split(CvDTreeNode* node)
{
	// update active mask
	int var_count = _active_var_mask->cols;
	RNG* rng = _recognizer->_rng;
	for(int vi=0; vi<var_count; vi++){
		uchar temp;
		int i1 = (*rng)(var_count);
		int i2 = (*rng)(var_count);
		CV_SWAP(_active_var_mask->data.ptr[i1],
			_active_var_mask->data.ptr[i2], temp);
	}

	// find best split
	CvDTreeSplit *bestSplit = 0;
	BPRTreeBestSplitFinder finder(this, node);
	finder.find(0, data->var_count);
	if(finder.bestSplit->quality > 0){
		bestSplit = data->new_split_cat(0, -1.0f);
		memcpy(bestSplit, finder.bestSplit, finder.splitSize);

		// copy and redirect best feature
		BPRecogPara& para = _recognizer->_para;
		int bestFeatureId = bestSplit->var_idx;
		_curFeatures[bestFeatureId].second = true;
	}
	
	string msg("split node: ");
	msg.append(LogCat::to_string(node->depth));
	msg.append(" (");
	msg.append(LogCat::to_string(node->sample_count));
	MSG(msg);

	return bestSplit;
}

void BPRTree::try_split_node( CvDTreeNode* node )
{
	_nodeCount++;
	CvDTree::try_split_node(node);
}

CvDTreeSplit* BPRTree::find_split_ord_class(CvDTreeNode* n, int vi, 
											float init_quality, CvDTreeSplit* _split, uchar* ext_buf)
{
	return CvDTree::find_split_ord_class(n, vi, init_quality, _split, ext_buf);
}
CvDTreeSplit* BPRTree::find_split_cat_class(CvDTreeNode* n, int vi, 
								   float init_quality, CvDTreeSplit* _split, uchar* ext_buf)
{
	return CvDTree::find_split_cat_class(n, vi, init_quality, _split, ext_buf);
}
CvDTreeSplit* BPRTree::find_split_ord_reg(CvDTreeNode* n, int vi, 
								 float init_quality, CvDTreeSplit* _split, uchar* ext_buf)
{
	return CvDTree::find_split_ord_reg(n, vi, init_quality, _split, ext_buf);
}
CvDTreeSplit* BPRTree::find_split_cat_reg(CvDTreeNode* n, int vi, 
								 float init_quality, CvDTreeSplit* _split, uchar* ext_buf)
{
	return CvDTree::find_split_cat_reg(n, vi, init_quality, _split, ext_buf);
}

BPRTreeBestSplitFinder::BPRTreeBestSplitFinder(BPRTree* _tree, CvDTreeNode* _node)
{
	tree = _tree;
	node = _node;
	splitSize = tree->get_data()->split_heap->elem_size;

	bestSplit = (CvDTreeSplit*)fastMalloc(splitSize);
	memset((CvDTreeSplit*)bestSplit, 0, splitSize);
	bestSplit->quality = -1;
	bestSplit->condensed_idx = INT_MIN;
	split = (CvDTreeSplit*)fastMalloc(splitSize);
	memset((CvDTreeSplit*)split, 0, splitSize);
}

BPRTreeBestSplitFinder::BPRTreeBestSplitFinder(const BPRTreeBestSplitFinder& finder, CvDTreeSplit* split)
{
	tree = finder.tree;
	node = finder.node;
	splitSize = tree->get_data()->split_heap->elem_size;

	bestSplit = (CvDTreeSplit*)fastMalloc(splitSize);
	memcpy((CvDTreeSplit*)(bestSplit), (const CvDTreeSplit*)finder.bestSplit, splitSize);
	split = (CvDTreeSplit*)fastMalloc(splitSize);
	memset((CvDTreeSplit*)split, 0, splitSize);
}

void BPRTreeBestSplitFinder::find(int var_start, int var_end)
{
	int n = node->sample_count;
	CvDTreeTrainData* data = tree->get_data();
	AutoBuffer<uchar> inn_buf(2*n*(sizeof(int) + sizeof(float)));

	for(int vi=var_start; vi<var_end; vi++){
		CvDTreeSplit *res;
		int ci = data->get_var_type(vi);
		if(node->get_num_valid(vi) <= 1 || tree->_active_var_mask->data.ptr[vi])
			continue;

		if(data->is_classifier){
			if(ci >= 0)
				res = tree->find_split_cat_class(node, vi, bestSplit->quality, split, (uchar*)inn_buf);
			else
				res = tree->find_split_ord_class(node, vi, bestSplit->quality, split, (uchar*)inn_buf);
		}else{
			if(ci >= 0)
				res = tree->find_split_cat_reg(node, vi, bestSplit->quality, split, (uchar*)inn_buf);
			else
				res = tree->find_split_ord_reg(node, vi, bestSplit->quality, split, (uchar*)inn_buf);
		}

		if(res && bestSplit->quality < split->quality)
			memcpy((CvDTreeSplit*)bestSplit, (CvDTreeSplit*)split, splitSize);
	}
}

void BPRTreeBestSplitFinder::join(BPRTreeBestSplitFinder& rhs)
{
	if(bestSplit->quality < rhs.bestSplit->quality)
		memcpy((CvDTreeSplit*)bestSplit, (CvDTreeSplit*)rhs.bestSplit, splitSize);
}

CvDTreeNode* BPRTree::predict(IplImage* img, CvPoint pixel,
							  vector<DepthFeature*>& fset) const
{
	int i;
	CvDTreeNode* node = root;

	if( !node )
		CV_Error( CV_StsError, "The tree has not been trained yet" );

	while( node->left )
	{
		CvDTreeSplit* split = node->split;
		int dir = 0;
		i = split->var_idx;
		float val = fset[i]->getValue(img, pixel);
		dir = val <= split->ord.c ? -1 : 1;

		node = dir < 0 ? node->left : node->right;
	}

	return node;
}
