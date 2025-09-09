#include <iostream>
 
using namespace std;
 
int main(){
    string dna;
    int counter = 1;
    int longest = 1;
    int i;
 
    cin >> dna;
    for (i = 0; i < dna.size(); i++){
        if (dna[i] == dna[i+1]){
            counter++;
            if (counter > longest){
                longest = counter;
            }
        }
        else{
            counter = 1;
        }
    }
    cout << longest;
}
