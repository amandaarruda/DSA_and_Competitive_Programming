#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; // test cases
    cin >> n;

    int s = 0; // score

    do{
        int q;
        cin >> q; // number of values

        vector<int> v(2*q); // vector to store the values

        int i;
        for (i = 0; i < 2*q; i++) {
            cin >> v[i];
        }

        sort(v.begin(), v.end());

        for (i = 0; i < 2*q; i++) {
            if (i % 2 == 0) {
                s += v[i]; // sum the values at even indices
            }
        }
        cout << s << "\n";
        s = 0; // reset score for next test case
    } while(--n);
}
