#include <iostream>
#include "MutationAlg/mp.h"

using namespace std;

int main() {

    string originalDNA = "ACGTACGTACGT"; // Initial DNA
    printf("Starting to load the model\n\n");
    DNA_Simulation dna_sim(originalDNA);
    printf("Starting simulation training\n\n");
    // Simple training model, initialization
    //trainModel(dna_sim.model);
    printf("Starting mutation prediction, original DNA: %s\n\n", originalDNA.c_str());
    // AI predicts mutation probability and replicates DNA
    string mutatedDNA = dna_sim.replicateDNA(32, 0.3); // 2 historical mutations, environmental factor 0.3
    cout << "Mutated DNA: " << mutatedDNA << endl;

    return 0;
}