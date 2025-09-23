#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t, n;
    cin >> t; // number of test cases

    while (t--) {
        cin >> n; // number of fields

        vector<long long> a(n);
        vector<long long> odd;
        long long total = 0;

        for (int i = 0; i < n; i++) {
            cin >> a[i];
            total += a[i];
            if (a[i] % 2 != 0) odd.push_back(a[i]);
        }

        if (odd.empty()) {
            cout << 0 << '\n';
            continue;
        }

        sort(odd.begin(), odd.end()); // crescente
        long long subtract = 0;
        int k = (int)odd.size();
        // somar os ⌊k/2⌋ menores ímpares para descartar
        for (int i = 0; i < k / 2; i++) subtract += odd[i];

        cout << (total - subtract) << '\n';
    }

    return 0;
}
