// Problem C — Outfit Ordeal
// Ideia: simular uma pilha de roupas com três operações:
//   - "put s": empilha s
//   - "get": desempilha topo ou imprime "empty" se vazia
//   - "iditarod": procura "snowcoat"; se achar, remove-o mantendo a ordem do resto
//                 e imprime "winner winner chicken dinner :)", senão "oopsimcold :("

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    vector<string> st; st.reserve(T);

    while (T--) {
        string cmd; 
        cin >> cmd;
        if (cmd == "put") {
            string s; cin >> s;
            st.push_back(s);
        } else if (cmd == "get") {
            if (st.empty()) {
                cout << "empty\n";
            } else {
                cout << st.back() << "\n";
                st.pop_back();
            }
        } else if (cmd == "iditarod") {
            // procura "snowcoat" (os nomes na pilha são distintos)
            const string target = "snowcoat";
            bool found = false;
            for (int i = (int)st.size()-1; i >= 0; --i) { // busca do topo para baixo (tanto faz, é único)
                if (st[i] == target) {
                    st.erase(st.begin() + i);
                    found = true;
                    break;
                }
            }
            if (found) cout << "winner winner chicken dinner :)\n";
            else       cout << "oopsimcold :(\n";
        }
    }
    return 0;
}

// Problem K — Romantic Glasses
// Ideia: existe subarray [l..r] em que soma dos ímpares == soma dos pares.
// Use prefixos D[i] = (soma dos a[j] em j<=i e j ímpar) - (soma dos a[j] em j<=i e j par).
// Existe subarray com igualdade <=> existem i<j com D[i]==D[j]. Basta checar repetição de D.

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t; 
    if (!(cin >> t)) return 0;
    while (t--) {
        int n; cin >> n;
        vector<long long> a(n+1);
        for (int i = 1; i <= n; ++i) cin >> a[i];

        unordered_set<long long> seen;
        seen.reserve(n*2);
        long long D = 0;        // D[0] = 0
        seen.insert(0);

        bool ok = false;
        for (int i = 1; i <= n; ++i) {
            if (i & 1) D += a[i]; else D -= a[i];
            if (seen.count(D)) { ok = true; break; }
            seen.insert(D);
        }
        cout << (ok ? "YES\n" : "NO\n");
    }
    return 0;
}
// Problem M — Cutting into Parts
// Ideia: com h cortes horizontais e v verticais, #partes = (h+1)*(v+1).
// Queremos exatamente n partes e minimizar h+v => escolher fatores a,b com a*b = n,
// minimizando (a-1)+(b-1) = a+b-2. Melhor par é o mais próximo de sqrt(n).

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n; 
    if (!(cin >> n)) return 0;

    long long ans = LLONG_MAX;
    for (long long d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            long long a = d, b = n / d;
            ans = min(ans, a + b - 2);
        }
    }
    cout << ans << "\n";
    return 0;
}

