#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <iomanip>

const std::string DB_PATH    = "C:/Users/meetj/Downloads/ObjectRecognition/data/training/objectdb.csv";
const std::string IMG_DIR    = "C:/Users/meetj/Downloads/ObjectRecognition/data/test_images/";
const std::string RES_DIR    = "C:/Users/meetj/Downloads/ObjectRecognition/results/";
const std::string MODEL_PATH = "C:/Users/meetj/Downloads/ObjectRecognition/data/resnet18-v2-7.onnx";

const std::vector<std::pair<std::string,std::string>> TRAIN_SET = {
    {"obj1_1.jpeg","object1"},{"obj1_2.jpeg","object1"},
    {"obj2_1.jpeg","object2"},{"obj2_2.jpeg","object2"},
    {"obj3_1.jpeg","object3"},{"obj3_2.jpeg","object3"},
    {"obj4_1.jpeg","object4"},{"obj4_2.jpeg","object4"},
    {"obj5_1.jpeg","object5"},{"obj5_2.jpeg","object5"}
};

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
    bool demoMode     = (argc > 1 && std::string(argv[1]) == "--demo");
    bool cnnMode      = (argc > 1 && std::string(argv[1]) == "--cnn");
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
                std::cout << "Stored: " << label << " fill=" << fv.percentFilled << std::endl;
            }
        }
        saveTrainingData(db, DB_PATH);
        std::cout << "Training complete! " << db.size() << " entries." << std::endl;
        return 0;
    }

    if (cnnMode) {
        std::cout << "=== CNN EMBEDDING MODE ===" << std::endl;
        cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);
        if (net.empty()) { std::cout << "Failed to load model!" << std::endl; return 1; }
        std::cout << "ResNet18 loaded." << std::endl;

        // Build CNN training embeddings
        std::vector<std::pair<std::string, cv::Mat>> cnnDB;
        for (auto& [fname, label] : TRAIN_SET) {
            cv::Mat src = cv::imread(IMG_DIR + fname);
            if (src.empty()) continue;
            cv::Mat binary  = applyThreshold(src);
            cv::Mat cleaned = applyMorphology(binary);
            cv::Mat labelViz;
            std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);
            if (!regions.empty()) {
                cv::Mat featDisplay;
                FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);
                cv::Mat embimg;
                prepEmbeddingImage(src, embimg,
                    (int)regions[0].centroid.x, (int)regions[0].centroid.y,
                    regions[0].theta,
                    regions[0].minE1, regions[0].maxE1,
                    regions[0].minE2, regions[0].maxE2);
                cv::Mat emb = getEmbedding(embimg, net);
                cnnDB.push_back({label, emb});
                std::cout << "CNN trained: " << label << std::endl;
            }
        }

        // Evaluate with CNN
        std::cout << "\n=== CNN EVALUATION ===" << std::endl;
        int confusion[5][5] = {};
        for (auto& [fname, trueLabel] : EVAL_SET) {
            cv::Mat src = cv::imread(IMG_DIR + fname);
            if (src.empty()) continue;
            cv::Mat binary  = applyThreshold(src);
            cv::Mat cleaned = applyMorphology(binary);
            cv::Mat labelViz;
            std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);
            std::string predicted = "unknown";
            if (!regions.empty()) {
                cv::Mat featDisplay;
                FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);
                cv::Mat embimg;
                prepEmbeddingImage(src, embimg,
                    (int)regions[0].centroid.x, (int)regions[0].centroid.y,
                    regions[0].theta,
                    regions[0].minE1, regions[0].maxE1,
                    regions[0].minE2, regions[0].maxE2);
                cv::Mat emb = getEmbedding(embimg, net);

                // Nearest neighbor by embedding distance
                double bestDist = 1e18;
                for (auto& [lbl, tEmb] : cnnDB) {
                    double d = embeddingDistance(emb, tEmb);
                    if (d < bestDist) { bestDist = d; predicted = lbl; }
                }
                cv::Mat result = src.clone();
                cv::rectangle(result, regions[0].boundingBox, cv::Scalar(0,255,0), 2);
                cv::putText(result, "CNN: " + predicted,
                    cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    predicted == trueLabel ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
                cv::imwrite(RES_DIR + "cnn_" + fname, result);
            }
            int ti = labelIndex(trueLabel), pi = labelIndex(predicted);
            if (ti >= 0 && pi >= 0) confusion[ti][pi]++;
            std::cout << fname << " -> " << predicted << " (" << trueLabel << ") "
                      << (predicted==trueLabel ? "CORRECT" : "WRONG") << std::endl;
        }

        std::cout << "\n=== CNN CONFUSION MATRIX ===" << std::endl;
        std::cout << std::setw(10) << " ";
        for (auto& l : LABELS) std::cout << std::setw(10) << l;
        std::cout << std::endl;
        int correct=0, total=0;
        for (int i = 0; i < 5; i++) {
            std::cout << std::setw(10) << LABELS[i];
            for (int j = 0; j < 5; j++) {
                std::cout << std::setw(10) << confusion[i][j];
                if (i==j) correct += confusion[i][j];
                total += confusion[i][j];
            }
            std::cout << std::endl;
        }
        std::cout << "\nCNN Accuracy: " << correct << "/" << total
                  << " = " << std::fixed << std::setprecision(1)
                  << (100.0*correct/total) << "%" << std::endl;
        return 0;
    }

    if (demoMode) {
        std::cout << "=== DEMO MODE - press any key to advance ===" << std::endl;
        for (auto& [fname, trueLabel] : EVAL_SET) {
            cv::Mat src = cv::imread(IMG_DIR + fname);
            if (src.empty()) continue;
            cv::Mat binary  = applyThreshold(src);
            cv::Mat cleaned = applyMorphology(binary);
            cv::Mat labelViz;
            std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);
            if (!regions.empty()) {
                cv::Mat featDisplay;
                FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);
                std::string predicted = classify(fv, db);
                cv::Mat result = src.clone();
                cv::rectangle(result, regions[0].boundingBox, cv::Scalar(0,255,0), 3);
                cv::putText(result, "Predicted: " + predicted, cv::Point(20,40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    predicted==trueLabel ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
                cv::putText(result, "True: " + trueLabel, cv::Point(20,80),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,0), 2);
                cv::imshow("1. Original", src);
                cv::imshow("2. Thresholded", binary);
                cv::imshow("3. Cleaned", cleaned);
                cv::imshow("4. Regions", labelViz);
                cv::imshow("5. Features", featDisplay);
                cv::imshow("6. Result", result);
            }
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    }

    // Normal evaluation
    std::cout << "=== EVALUATION MODE ===" << std::endl;
    int confusion[5][5] = {};
    for (auto& [fname, trueLabel] : EVAL_SET) {
        cv::Mat src = cv::imread(IMG_DIR + fname);
        if (src.empty()) continue;
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
                cv::Point(regions[0].boundingBox.x, regions[0].boundingBox.y-10),
                cv::FONT_HERSHEY_SIMPLEX, 1.2,
                predicted==trueLabel ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
            cv::imwrite(RES_DIR + "classified_" + fname, result);
        }
        int ti = labelIndex(trueLabel), pi = labelIndex(predicted);
        if (ti >= 0 && pi >= 0) confusion[ti][pi]++;
        std::cout << fname << " -> predicted=" << predicted
                  << " true=" << trueLabel
                  << (predicted==trueLabel ? " CORRECT" : " WRONG") << std::endl;
    }
    std::cout << "\n=== CONFUSION MATRIX ===" << std::endl;
    std::cout << std::setw(10) << " ";
    for (auto& l : LABELS) std::cout << std::setw(10) << l;
    std::cout << std::endl;
    int correct=0, total=0;
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << LABELS[i];
        for (int j = 0; j < 5; j++) {
            std::cout << std::setw(10) << confusion[i][j];
            if (i==j) correct += confusion[i][j];
            total += confusion[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "\nAccuracy: " << correct << "/" << total
              << " = " << std::fixed << std::setprecision(1)
              << (100.0*correct/total) << "%" << std::endl;
    return 0;
}
