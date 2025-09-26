#include <iostream>
#include <vector>
using namespace std;
 
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
 
    int a; // number of elements
    cin >> a;
 
    long long counter = 0; // number of moves
 
    vector<long long> v(a);
 
    int i;
 
    for (int i = 0; i < a; i++) {
        cin >> v[i];
    }
 
    for (i = 0; i < a; i++) {
        if (i != 0 && v[i] < v[i-1]) {
            counter += (v[i-1] - v[i]);
            v[i] = v[i-1];
        }
    }
 
    cout << counter;
    
}
