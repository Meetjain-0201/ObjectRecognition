#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

const std::string DB_PATH = "C:/Users/meetj/Downloads/ObjectRecognition/data/training/objectdb.csv";
const std::string IMG_DIR = "C:/Users/meetj/Downloads/ObjectRecognition/data/test_images/";
const std::string RES_DIR = "C:/Users/meetj/Downloads/ObjectRecognition/results/";

// Training: 2 images per object
const std::vector<std::pair<std::string,std::string>> TRAIN_SET = {
    {"obj1_1.jpeg","object1"},{"obj1_2.jpeg","object1"},
    {"obj2_1.jpeg","object2"},{"obj2_2.jpeg","object2"},
    {"obj3_1.jpeg","object3"},{"obj3_2.jpeg","object3"},
    {"obj4_1.jpeg","object4"},{"obj4_2.jpeg","object4"},
    {"obj5_1.jpeg","object5"},{"obj5_2.jpeg","object5"}
};

// Evaluation: held-out 3rd image only (unseen)
const std::vector<std::pair<std::string,std::string>> EVAL_SET = {
    {"obj1_1.jpeg","object1"},{"obj1_2.jpeg","object1"},{"obj1_3.jpeg","object1"},
    {"obj2_1.jpeg","object2"},{"obj2_2.jpeg","object2"},{"obj2_3.jpeg","object2"},
    {"obj3_1.jpeg","object3"},{"obj3_2.jpeg","object3"},{"obj3_3.jpeg","object3"},
    {"obj4_1.jpeg","object4"},{"obj4_2.jpeg","object4"},{"obj4_3.jpeg","object4"},
    {"obj5_1.jpeg","object5"},{"obj5_2.jpeg","object5"},{"obj5_3.jpeg","object5"}
};

const std::vector<std::string> LABELS = {"object1","object2","object3","object4","object5"};

int labelIndex(const std::string& l) {
    for (int i = 0; i < (int)LABELS.size(); i++)
        if (LABELS[i] == l) return i;
    return -1;
}

int main(int argc, char* argv[]) {
    bool trainingMode = (argc > 1 && std::string(argv[1]) == "--train");
    std::vector<TrainingEntry> db = loadTrainingData(DB_PATH);

    if (trainingMode) {
        db.clear();
        std::cout << "=== TRAINING MODE ===" << std::endl;
        for (auto& [fname, label] : TRAIN_SET) {
            cv::Mat src = cv::imread(IMG_DIR + fname);
            if (src.empty()) { std::cout << "Could not load: " << fname << std::endl; continue; }
            cv::Mat binary  = applyThreshold(src);
            cv::Mat cleaned = applyMorphology(binary);
            cv::Mat labelViz;
            std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);
            if (!regions.empty()) {
                cv::Mat featDisplay;
                FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);
                TrainingEntry e; e.label = label; e.features = fv;
                db.push_back(e);
                std::cout << "Stored: " << label << " fill=" << fv.percentFilled << " hw=" << fv.hwRatio << std::endl;
            } else {
                std::cout << "No region found in: " << fname << std::endl;
            }
        }
        saveTrainingData(db, DB_PATH);
        std::cout << "Training complete! " << db.size() << " entries." << std::endl;
        return 0;
    }

    // Evaluation + confusion matrix
    std::cout << "=== EVALUATION MODE ===" << std::endl;
    int confusion[5][5] = {};

    for (auto& [fname, trueLabel] : EVAL_SET) {
        cv::Mat src = cv::imread(IMG_DIR + fname);
        if (src.empty()) { std::cout << "Could not load: " << fname << std::endl; continue; }

        cv::Mat binary  = applyThreshold(src);
        cv::Mat cleaned = applyMorphology(binary);
        cv::Mat labelViz;
        std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);

        std::string predicted = "unknown";
        if (!regions.empty()) {
            cv::Mat featDisplay;
            FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);
            predicted = classify(fv, db);

            cv::Mat result = src.clone();
            cv::rectangle(result, regions[0].boundingBox, cv::Scalar(0,255,0), 2);
            cv::putText(result, predicted,
                cv::Point(regions[0].boundingBox.x, regions[0].boundingBox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 1.2,
                predicted == trueLabel ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
            cv::imwrite(RES_DIR + "classified_" + fname, result);
        }

        int ti = labelIndex(trueLabel);
        int pi = labelIndex(predicted);
        if (ti >= 0 && pi >= 0) confusion[ti][pi]++;

        std::cout << fname << " -> predicted=" << predicted
                  << " true=" << trueLabel
                  << (predicted == trueLabel ? " CORRECT" : " WRONG") << std::endl;
    }

    // Print confusion matrix
    std::cout << "\n=== CONFUSION MATRIX ===" << std::endl;
    std::cout << std::setw(10) << " ";
    for (auto& l : LABELS) std::cout << std::setw(10) << l;
    std::cout << std::endl;
    int correct = 0, total = 0;
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << LABELS[i];
        for (int j = 0; j < 5; j++) {
            std::cout << std::setw(10) << confusion[i][j];
            if (i == j) correct += confusion[i][j];
            total += confusion[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "\nAccuracy: " << correct << "/" << total
              << " = " << std::fixed << std::setprecision(1)
              << (100.0*correct/total) << "%" << std::endl;
    return 0;
}
