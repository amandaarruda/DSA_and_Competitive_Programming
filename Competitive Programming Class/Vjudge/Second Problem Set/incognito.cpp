#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;

    while (t--) {
        int n;
        cin >> n;

        map<string,int> disguises;
        for (int i = 0; i < n; i++) {
            string c, k;
            cin >> c >> k;
            disguises[k]++;
        }

        long long total = 1;
        for (auto& [k, count] : disguises) {
            total *= (count + 1);
        }

        cout << (total - 1) << "\n";
    }
    return 0;
}
