#include "objectrec.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>

void saveTrainingData(const std::vector<TrainingEntry>& db, const std::string& path) {
    std::ofstream f(path);
    for (auto& e : db) {
        f << e.label << ","
          << e.features.percentFilled << ","
          << e.features.hwRatio << ","
          << e.features.hu1 << ","
          << e.features.hu2 << ","
          << e.features.hu3 << "\n";
    }
    std::cout << "Saved " << db.size() << " entries to " << path << std::endl;
}

std::vector<TrainingEntry> loadTrainingData(const std::string& path) {
    std::vector<TrainingEntry> db;
    std::ifstream f(path);
    if (!f.is_open()) return db;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        TrainingEntry e;
        std::getline(ss, e.label, ',');
        std::getline(ss, tok, ','); e.features.percentFilled = std::stod(tok);
        std::getline(ss, tok, ','); e.features.hwRatio       = std::stod(tok);
        std::getline(ss, tok, ','); e.features.hu1           = std::stod(tok);
        std::getline(ss, tok, ','); e.features.hu2           = std::stod(tok);
        std::getline(ss, tok, ','); e.features.hu3           = std::stod(tok);
        db.push_back(e);
    }
    std::cout << "Loaded " << db.size() << " entries from " << path << std::endl;
    return db;
}

// Scaled Euclidean distance
static double scaledDist(const FeatureVector& a, const FeatureVector& b,
                          const std::vector<double>& stdevs) {
    std::vector<double> diff = {
        a.percentFilled - b.percentFilled,
        a.hwRatio       - b.hwRatio,
        a.hu1           - b.hu1,
        a.hu2           - b.hu2,
        a.hu3           - b.hu3
    };
    double dist = 0;
    for (int i = 0; i < 5; i++) {
        double s = stdevs[i] > 1e-6 ? stdevs[i] : 1.0;
        dist += (diff[i]/s) * (diff[i]/s);
    }
    return sqrt(dist);
}

std::string classify(const FeatureVector& fv, const std::vector<TrainingEntry>& db) {
    if (db.empty()) return "unknown";

    // Compute stdev for each feature across db
    std::vector<std::vector<double>> feats(5, std::vector<double>(db.size()));
    for (int j = 0; j < (int)db.size(); j++) {
        feats[0][j] = db[j].features.percentFilled;
        feats[1][j] = db[j].features.hwRatio;
        feats[2][j] = db[j].features.hu1;
        feats[3][j] = db[j].features.hu2;
        feats[4][j] = db[j].features.hu3;
    }
    std::vector<double> stdevs(5);
    for (int i = 0; i < 5; i++) {
        double mean = std::accumulate(feats[i].begin(), feats[i].end(), 0.0) / feats[i].size();
        double var  = 0;
        for (double v : feats[i]) var += (v-mean)*(v-mean);
        stdevs[i] = sqrt(var / feats[i].size());
    }

    // Nearest neighbor
    double bestDist = 1e18;
    std::string bestLabel = "unknown";
    for (auto& e : db) {
        double d = scaledDist(fv, e.features, stdevs);
        if (d < bestDist) { bestDist = d; bestLabel = e.label; }
    }
    return bestLabel;
}
