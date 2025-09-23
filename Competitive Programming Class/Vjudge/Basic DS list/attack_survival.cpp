#include <iostream>
#include <vector>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, k, q;
    cin >> n >> k >> q; // n players, k inicial points, q questions

    vector<int> a(n); // points of each player

    for (int i = 0; i < q; i++){
        int winner;
        cin >> winner;
        winner--; 

        a[winner]++;
    }
    
    for (int i = 0; i < n; i++){
        if (a[i] + k - q > 0){
            cout << "Yes" << endl;
        } else {
            cout << "No" << endl;
        }
    }
}
