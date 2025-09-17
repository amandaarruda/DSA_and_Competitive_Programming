#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    int q;
    cin >> n >> q;

    vector<int> v(n);
    for (int i = 0; i < n; i++) {
        cin >> v[i];
    }

    do{
        int t; // target
        cin >> t;

        auto it = lower_bound(v.begin(), v.end(), t);

        if (it != v.end() && *it == t) {
            cout << it - v.begin();
        }else{
            cout << "-1";
        }

        cout << "\n";

    } while (--q);
}
