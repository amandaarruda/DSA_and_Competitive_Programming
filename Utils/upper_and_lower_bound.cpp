#include <bits/stdc++.h>
using namespace std;

int main() {
    multiset<int> s = {1, 2, 4, 4, 5, 7};

    cout << "Conjunto: ";
    for (int v : s) cout << v << " ";
    cout << "\n\n";

    int x = 4;

    // lower_bound
    auto it1 = s.lower_bound(x);
    if (it1 != s.end())
        cout << "lower_bound(" << x << ") = " << *it1 << "\n";
    else
        cout << "lower_bound(" << x << ") = fim\n";

    // upper_bound
    auto it2 = s.upper_bound(x);
    if (it2 != s.end())
        cout << "upper_bound(" << x << ") = " << *it2 << "\n";
    else
        cout << "upper_bound(" << x << ") = fim\n";

    return 0;
}
