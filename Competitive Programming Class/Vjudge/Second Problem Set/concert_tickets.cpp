#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    multiset<int> tickets;
    for (int i = 0; i < n; i++) {
        int price;
        cin >> price;
        tickets.insert(price);
    }

    for (int i = 0; i < m; i++) {
        int max_price;
        cin >> max_price;

        // quero checar qual o prox. ingresso maior do que o que eu posso pagar
        auto it = tickets.upper_bound(max_price);

        // se encontrar o começo do multiset
        if (it == tickets.begin()) {
            // não existe nenhum
            cout << -1 << "\n";
        } else {
            // volto uma posição pra pegar o maior ingresso que seja = ou menor que t
            it--;
            cout << *it << endl;
            tickets.erase(it);
        }
    }

    return 0;
}
