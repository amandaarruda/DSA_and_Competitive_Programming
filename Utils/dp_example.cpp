#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; cin >> n;

    vector<int> v(n+1, 100);
    v[0] = 0;

    for (int i = 1; i <= n; i++){
        int x = i;
        
        while (x > 0) {
            int d = x % 10;
            x /= 10;
            if (d > 0) {
                v[i] = min(v[i], v[i - d] + 1);
            }
        }
    }

    cout << v[n];

    return 0;
}
