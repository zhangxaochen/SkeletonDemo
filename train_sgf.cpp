#include "train_sgf.h"

bool train_SGF::train(int data_source, string data_path)
{
    string name = "training samples";
    DepthSampleGenerator* dsg = new DepthSampleGenerator(name);

    //set data path
    dsg->setDataSource(data_source,data_path);
    
    //generate data from images  xxx.bmp not .png
    //need a sample_info in the image path folder
    bool data_generated = dsg->generateDataFromFile();
    if (!data_generated)
    {
        return false;
    }

    //set decision tree paramenters
    BPRecogPara para;

    //create BPRecognizer instance
    BPRecognizer* bpr = new BPRecognizer();
    bpr->setPara(para);
    bpr->setSampleGenerator(dsg);
    
    //create BPRtree instance
    BPRTree* tree = new BPRTree();

    //start training
    bool trained = bpr->train();
    bool saved = bpr->save(data_path);
    //bpr->test(true);
    return trained;
}